# Eight-rank phase-field AMG tuning

## Selected configuration

This note records the iterative-solver tuning. The executable now defaults to
[distributed MUMPS direct solves](phase-field-direct-solver.md); select `-ls gmres`
to reproduce these measurements. With that selection it uses separate GMRES/BoomerAMG instances for displacement
and phase. Displacement defaults to MFEM's `SetSystemsOptions(2)`, appropriate
to the existing two-component `Ordering::byVDIM` space; phase retains scalar
AMG. On the measured problem, this was faster than scalar AMG or either tested
elasticity-interpolation configuration. It is a measured default, not a claim
of optimality for all meshes or stages of crack propagation.

The user's command remains valid from the executable directory:

```bash
mpiexec -n 8 ./pPhaseField_shear -ls gmres -rp 4 -il 1
```

Use the Release executable for performance. Other configurations remain
available through `-uamg` / `--displacement-amg`:

| Value | Displacement preconditioner |
| --- | --- |
| `systems` (default) | MFEM systems AMG, two functions, no aggressive coarsening, strength threshold 0.5 |
| `scalar` | Previous MFEM scalar AMG settings |
| `elasticity` | MFEM elasticity options, including rotational near-nullspace interpolation and interpolation refinement |
| `elasticity-no-refine` | Same elasticity options without interpolation refinement |

These are MFEM's existing configuration bundles. The installed implementation
of `SetSystemsOptions` was inspected in MFEM 4.9.1 `linalg/hypre.cpp`.
[MFEM's API documentation](https://docs.mfem.org/4.9/classmfem_1_1HypreBoomerAMG.html)
describes both systems and elasticity options. Hypre explains the distinction
between unknown-based systems coarsening and near-nullspace interpolation in
[AMG for systems of PDEs](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html#amg-for-systems-of-pdes).
The selection here comes from the measurements below, rather than an assumption
that elasticity interpolation is always faster.

The linear relative tolerance remains `1e-10` (double precision), restart 50,
and maximum 2000 iterations per solve. GMRES uses its preconditioned residual
for stopping. `-il 1` additionally reports the true residual
`||b-Ax||/||b||`, block identity, iterations, convergence status, and timings.
True residuals need not equal the preconditioned tolerance. No physical
parameters, phase bounds, nonlinear tolerances, or acceptance rules were relaxed.

`SetBlockSolvers` is an explicit borrowed-solver API in `NewtonForPhaseField`.
`SetSolver` restores the original single-solver behavior. A regression test
checks block dispatch and restoration of the shared-solver contract. Each
preconditioner rebuilds for its new tangent; there is no hierarchy reuse across
changed matrices. The existing MPI form, local true-DOF ownership, Hypre ParCSR
operators, and MPI capability gate are unchanged.

## Reproducible workload

Measurements on 2026-09-05 used base revision
`528c91c09c4d0d6a95a1396eeb345a14678ce5d0` plus the uncommitted phase-field
changes, GCC 15.2.0, Release `-O3 -DNDEBUG`, MFEM 4.9.1, Hypre 3.0.0, double
precision, and an Intel i9-10900KF (10 physical cores, 20 hardware threads).
MFEM OpenMP/CUDA were disabled. Eight MPI ranks were bound to cores with one
OpenMP thread per process. Experiments ran sequentially to avoid contention.

The mesh was `data/crack_square2d_quad.msh`, H1 order 1, serial refinement 3,
parallel refinement 4: 65,536 elements, 132,354 displacement true DOFs and
66,177 phase true DOFs. Material parameters and boundary conditions are those
in [the model document](phase-field-fracture.md). There is no random input or
external solver option file.

After configuring/building with the Release preset, run from a disposable
working directory using absolute paths:

```bash
mpiexec -n 8 --bind-to core -x OMP_NUM_THREADS=1 \
  /path/to/mfemMechanics/build/release/bin/pPhaseField_shear \
  -m /path/to/mfemMechanics/data/crack_square2d_quad.msh \
  -ls gmres -rs 3 -rp 4 -il 1 -uamg systems \
  -disp 1e-9 -dt 1 -dt-max 1 -steps 1 -no-vis
```

Repeat with each `-uamg` value. This is a completed small-load increment, with
final pseudo-time 1, not a full default fracture history. It reproduced the
reported 1150-iteration first displacement solve. Both blocks converged with
one nonlinear sweep. Initial exploratory runs preceded three measured repeats.
Logs, GNU `time -v` records and the baseline executable were retained under
`/tmp/phase-amg-tuning`; temporary files are not required by the implementation.

## Results

Medians of three runs on the requested mesh, in seconds:

| Displacement AMG | Displacement iterations | AMG setup | Krylov solve | Setup + solve | Whole run | GNU time maximum RSS, KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar | 1150 | 0.0171 | 6.1657 | 6.1827 | 6.87 | 81916 |
| Systems | 14 | 0.0340 | 0.1151 | 0.1492 | 0.83 | 81588 |
| Elasticity | 14 | 0.0791 | 0.1524 | 0.2317 | 0.87 | 84880 |
| Elasticity without refinement | 14 | 0.0704 | 0.1524 | 0.2229 | 0.92 | 85120 |

Systems AMG reduced displacement setup-plus-solve time by about 41× and whole
small-run time by about 8.3×. Phase solves retained 13 iterations. First-solve
true relative residuals were `5.59e-10` for scalar and `1.82e-10` for systems;
both met the same preconditioned stopping criterion. Timings include the
original assembly work in the whole-run measurement; no assembly speedup is
claimed.

Per-solve timings use `MPI_Wtime`; AMG's actual lazy setup call is timed, and
Krylov time is the enclosing GMRES duration minus setup on each rank. The
reported values are separate maximum-rank reductions, so setup and solve
maxima need not sum exactly to the reported total. Diagnostic residual
recomputation/reductions are outside that total but inside whole-run timing.
With info level 0, timing and diagnostic residual calculations are disabled.
GNU time's maximum RSS is its maximum process/child RSS statistic, not summed
memory across ranks.

A second mesh size (`-rp 3`, still eight ranks) gave 437 versus 12 displacement
iterations and setup-plus-solve times 0.2765 versus 0.0218 seconds for scalar
versus systems. This is a robustness check at a second size, not a parallel
scaling study.

## Numerical comparison at larger load

A second completed workload used the requested mesh and eight ranks with
`-disp 1e-6 -dt 0.5 -dt-max 0.5 -steps 2 -vis`. Scalar, systems and elasticity
AMG all converged with two nonlinear sweeps in the first increment and three
in the second. Displacement solve counts were unchanged; scalar took
1140–1234 iterations per solve, systems 12–14, and elasticity 13–14. Phase
solves remained at 13 iterations.

Scalar and systems reaction CSVs agreed at printed precision: 22957.6 N/m at
`5e-7 m` and 45904.3 N/m at `1e-6 m`. Comparing the same final exported VTK
point samples across all ranks gave relative Euclidean differences of
`3.33e-14` for displacement, `1.82e-10` for phase, and `4.25e-13` for stress.
These are differences of sampled output arrays, not FE-integrated error norms.
Maximum phase was approximately `0.00355174`; the phase-bound checks passed.

These workloads cover initial loading and small damage. The best configuration
can change once a crack develops and residual stiffness dominates. The full
100,000-attempt default history, severe-damage robustness, and scaling across
rank counts have not been benchmarked. Reassess using the reported true
residuals, nonlinear convergence, and setup/solve times if iterations rise
later; do not interpret this tuning as proof of uniform performance throughout
fracture.

## Final verification

Both full builds completed with `cmake --build --preset debug --parallel 4`
and `cmake --build --preset release --parallel 4`. The final commands
`ctest --test-dir build/debug --output-on-failure` and
`ctest --test-dir build/release --output-on-failure` each passed **107/107**,
including the new block-solver regression and serial/one-rank/two-rank shear
smokes. MPI tests ran outside the sandbox to permit PMIx sockets.
Touched C++ passed clang-format verification and `git diff --check` passed.
Single-precision verification was not performed. Changes remain uncommitted;
the unrelated SemismoothContact stash was preserved.

# Distributed direct solves for phase-field shear

`pPhaseField_shear` defaults to `-ls direct` (`--linear-solver direct`), using
separate MFEM MUMPS solvers for displacement and phase. This explicitly honors
the request to replace GMRES/AMG, rather than silently choosing an iterative
solver on some installations. MFEM must have MPI and MUMPS enabled. Without
MUMPS the example still builds, but direct selection (including the default)
fails before mesh loading with instructions to enable MUMPS or use `-ls gmres`.
The CMake target remains MPI-gated; direct tests are additionally MUMPS-gated.

`-ls gmres` retains the previous tolerances, restart, iteration limit, and all
`-uamg` settings described in [AMG tuning](phase-field-amg-tuning.md).
`-uamg` has no effect on direct solves. No material parameters, refinement
defaults, phase bounds, continuation settings, nonlinear tolerances, or the
default `-ni 12` sweep budget have changed.

## Operator and lifetime contract

The existing parallel path is retained: `ParMesh`, `ParFiniteElementSpace`,
`ParGridFunction`, and `ParBlockNonlinearForm`. The form assembles distributed
`HypreParMatrix` diagonal gradient blocks, and each rank passes its **owned
true-DOF** right-hand side and correction to MUMPS. This is not serial UMFPack
on a gathered matrix. The serial example remains unchanged.

Each block update calls `SetOperator` on the current tangent immediately before
`Mult`. MFEM's MUMPS adapter analyzes and factorizes in `SetOperator`; symbolic
reuse is explicitly disabled, so neither matrix values nor sparsity are assumed
unchanged between updates. General (`UNSYMMETRIC`) mode factors the full matrix
without assuming SPD or discarding a triangle. This conservative choice avoids
imposing extra symmetry/definiteness contracts on the assembled operator.
MUMPS's default automatic ordering is retained; no approximate BLR option is
enabled. MFEM handles MUMPS factorization errors; nonlinear convergence and phase
bounds remain checked by the existing driver.

The nonlinear solver borrows both block solvers. They outlive it and are
destroyed before the forms/spaces and MPI finalization. The diagnostic wrapper
borrows the tangent only for the immediate solve, before another gradient
assembly can invalidate it. All norm and timing collectives run on all ranks;
only printing is rank-zero-only.

Implementation contracts were checked against installed MFEM 4.9.1
`linalg/mumps.hpp` and `linalg/mumps.cpp` (`SetOperator`, `Mult`, `SetParameters`).
The installed MUMPS is 5.8.2, for which MFEM uses distributed RHS/solution
storage. Older MFEM adapters for MUMPS before 5.3 may internally centralize RHS
vectors; the example introduces no gather or serial matrix factorization.

## Diagnostics and limitations

`-il 1` reports `MUMPS solve` with block identity, true global Euclidean relative
residual `||b-Ax||/||b||`, analysis-plus-factorization time, solve time, and their
total. Each time is a separate maximum over ranks, so the printed maxima need
not add exactly. Residual recomputation and reductions are outside these times.
A zero RHS reports zero for a zero residual and infinity otherwise. Direct
reports deliberately omit Krylov iterations, preconditioned norms, and a
misleading iterative convergence flag. `-il 0` disables the extra diagnostics.
GMRES reports retain their original fields.

Direct solves remove Krylov iteration error/cost, **not the staggered
displacement/phase contraction**. The outer method and `-ni` budget still govern
coupled convergence; see [stagnation investigation](phase-field-stagnation.md).
Factorization fill can require substantially more memory than AMG. No speedup,
large-mesh memory bound, or cure for full-fracture nonconvergence is claimed.

## Reproduce and verify

From the repository root:

```bash
cmake --preset debug
cmake --build build/debug --parallel 4
ctest --test-dir build/debug --output-on-failure
cmake --preset release
cmake --build build/release --target pPhaseField_shear --parallel 4
ctest --test-dir build/release -R 'PhaseField' --output-on-failure

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpiexec -n 2 \
  "$PWD/build/release/bin/pPhaseField_shear" \
  -m "$PWD/data/crack_square2d_quad.msh" -ls direct -il 1 \
  -rs 4 -rp 0 -disp 1e-6 -tf 1 -dt 0.5 -dt-max 0.5 -steps 2 -no-vis
```

Repeat with `-ls gmres` and one rank. For reaction comparison, use absolute
executable/mesh paths from separate disposable working directories and replace
`-no-vis` by `-vis`; compare `p_phase_field_force.csv`. To switch an existing
production invocation, add `-ls direct` (or omit `-ls`); retain its other options.

Verification on 2026-09-05 used revision
`528c91c09c4d0d6a95a1396eeb345a14678ce5d0` plus the existing dirty workspace and
this change, the workspace Debug/Release configurations, GCC 15.2.0, MFEM 4.9.1
double precision, MPI and MUMPS 5.8.2 enabled, MFEM CUDA/OpenMP disabled.
The comparison used one OpenMP/BLAS thread per rank, no explicit rank binding,
H1 order 1, 1024 elements, 2210 displacement and 1105 phase global true DOFs,
domain attribute 1 and bottom/top boundary attributes 11/12. Other parameters
and SI units are those in [the formulation](phase-field-fracture.md); no random
seed or external solver options are involved.

- Full Debug build and CTest: **125/125 passed**.
- Release phase-field CTest selection: **28/28 passed**.
- Both solvers, one and two ranks, reached pseudo-time 1 in two accepted
  increments (two then three sweeps), exercising five factorizations/solves per
  block and unchanged phase-bound checks.
- Reaction CSVs agreed at existing printed precision: **23266.7 N/m** at
  `5e-7 m`, **46522.7 N/m** at `1e-6 m`. This is a reaction comparison, not a
  full-field error estimate.
- Maximum true relative linear residual over both ranks/counts and blocks:
  direct **4.54e-15** (rounded up), GMRES **4.39e-10** (rounded up). The comparison
  required each below `1e-8` and every GMRES solve to report convergence.
- Two-rank checks also passed for omitted `-ls` (MUMPS selected), and GMRES
  `scalar`, `elasticity`, and `elasticity-no-refine` AMG options at `-rs 0 -rp 0
  -disp 1e-9 -dt 1 -dt-max 1 -steps 1 -no-vis -il 1`.

The initial `-rs 0` repeated-solve test at `1e-6 m` failed the existing
phase-bound guard (minimum about `-1.28e-5` after the first increment). The
regression instead uses `-rs 4`; bounds/tolerances were not relaxed. This
underscores that direct factorization does not enforce a discrete maximum
principle. Full fracture histories, single precision, and an actual MFEM
installation without MUMPS were not tested.

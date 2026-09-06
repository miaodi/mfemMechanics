# Phase-field shear: staggered iteration budget investigation

The investigation below predates the default change to MUMPS. Add `-ls gmres`
to its MPI commands to reproduce the iterative baseline. See
[direct solver selection](phase-field-direct-solver.md) for the new default.
Direct block solves do not remove the staggered contraction discussed here.

## Status and scope

The MPI shear example now accepts `-ni N` / `--nonlinear-iterations N` to
configure the maximum block-Newton sweeps per continuation attempt. The default
remains 12 for compatibility. This is a work-budget control, **not a tolerance
change or a new nonlinear method**. It also affects the existing adaptive step
controller, which targets 80% of that budget when choosing the next increment.

A budget of 1000 passed the reported 10.673% stagnation location on the reported
mesh/refinement. This is not a claim that the complete fracture simulation is
fixed: the full-load experiment subsequently failed the existing phase-bounds
guard with a maximum nodal phase of about 1.01605. The guard is retained and now
reports its minimum, maximum, and tolerance. No phase clipping, mass lumping,
constitutive regularization, or relaxed convergence tolerance was introduced.

## Why accurate linear solves do not settle this problem

`NewtonForPhaseField` applies one displacement Newton correction at fixed phase,
then a phase correction at the updated displacement. It evaluates both
residuals at the current common state after either correction. Each goal is
`max(abs_tol, rel_tol * first_nonzero_block_norm)` for that nonlinear attempt.
The two residuals have different physical dimensions and are tested separately.

For the current small-strain AT2 formulation, fixed displacement fixes
`H_trial = max(H_committed, psi_plus(u))`. The phase residual is then affine in
phase. Solving it accurately can leave a roundoff-sized phase residual while
the changed degradation disturbs displacement equilibrium. A final
displacement-only correction can satisfy both goals without solving phase
again; its nonzero phase residual is acceptable only if it is below its goal.
The solver log now prints both actual goals.

The following is an original local linearization argument, not a new material
model. Write the block Jacobian as `[A B; C D]`. Exact block Gauss-Seidel gives

```text
e_u(next)   = -A^-1 B e_phi(current)
e_phi(next) =  D^-1 C A^-1 B e_phi(current).
```

Accurate diagonal solves do not make the spectral radius of
`D^-1 C A^-1 B` small. For example, the SPD scalar-block matrix
`[1 a; a 1]`, with right-hand side `(1, 0)` and zero initial state, has
`r_phi = 0` and `r_u = -a^(2*n)` after `n` sweeps. This explains slow coupling
convergence without any nonlinear displacement or spectral-tangent defect.
It does not measure the actual shear Jacobian's spectral radius.

Repeated cutbacks are not guaranteed to cure this contraction: the controller
can shrink even accepted increments when their sweep count exceeds its target,
and an accepted state can retain phase residual close to the absolute goal.
Increasing the budget changes both the allowed nonlinear work and the accepted
increment sequence. It must therefore be recorded as part of the experiment.

The material's spectral divided-difference tangent and all four assembled
Jacobian blocks pass their existing directional-difference tests. The experiments
below support slow coupling as the cause of the observed iteration-budget
bottleneck; they do not prove global convergence or rule out every untested
material state. An inner displacement solve was not added: extra displacement
Newton iterations do not in general remove the coupling contraction above.

## Reproduction environment

- Repository base: `528c91c09c4d0d6a95a1396eeb345a14678ce5d0`, with the existing
  user-owned dirty phase-field changes plus the diagnostics, option, and tests
  described here. The commit alone does not reproduce this workspace.
- Existing `build/release` configuration was reused, not reconfigured:
  `/usr/bin/c++`, `-O3 -DNDEBUG`, installed MFEM at
  `/home/miaodi/repo/mfem/install/release`.
- MFEM reports version integer 40901 and Git string
  `heads/master-0-g3b6b6f295f569eee066fec48037003297d505955-dirty`;
  double precision, MPI/METIS enabled, MFEM OpenMP and CUDA disabled.
- Mesh: `data/crack_square2d_quad.msh`; domain attribute 1, bottom/top
  attributes 11/12; Q1, serial refinement 3 and parallel refinement 2,
  no local refinement. Global true sizes: 8514 displacement, 4257 phase.
- Existing physical defaults: Miehe spectral split, `E=210e9`, `nu=0.3`,
  `Gc=2700`, length scale `1.5e-5`, residual stiffness `1e-9`. Initial, maximum,
  and minimum continuation increments are `1e-6`, `1e-3`, and `1e-14`, with
  at most 100000 attempts. No random input.
- Existing AMG/GMRES defaults retained: displacement systems AMG, separate phase
  AMG, restart 50, at most 2000 iterations, relative tolerance `1e-10` for both
  linear solvers. The much smaller phase residuals in the log are achieved
  residuals, not a separate configured phase tolerance.
- Host-only assembly; 1 or 4 MPI ranks as specified, no explicit thread override.
  No performance or scaling claim is made.

## Commands and observations

Build from the repository root:

```bash
cmake --build build/release --target pPhaseField_shear phase_field_test --parallel 2
```

The reported command was `mpiexec -n 4 ./pPhaseField_shear -rp 2 -il 1`.
The investigative baseline used absolute paths and disabled output:

```bash
# Working directory: /tmp/opencode/phase-stagnation
timeout 180s mpiexec -n 4 \
  /home/miaodi/repo/mfemMechanics/build/release/bin/pPhaseField_shear \
  -m /home/miaodi/repo/mfemMechanics/data/crack_square2d_quad.msh \
  -rp 2 -il 1 -no-vis > baseline.log 2>&1
```

`-no-vis` disables ParaView and force-curve output, not the phase-bounds check.
The baseline was time-limited at about 9.998%, reproducing the slow-down but
**not independently reproducing the final minimum-increment abort**. Near its
end, phase-completed sweeps reduced displacement residual approximately as
`2.16409, 1.57014, 1.13864, 0.825518, 0.598408, 0.433738` while phase residuals
were near roundoff. A displacement-only correction then returned
`(3.62837e-8, 3.4101e-7)` against goals `(3.69784e-5, 4.66294e-7)`.

The same command with `-ni 100` was time-limited at 180 seconds near 10.659%.
Displacement residual was still contracting (roughly 0.975 per sweep), not
being held constant by inaccurate diagonal solves.

The same command with `-ni 1000` passed 10.710% after a 305-sweep increment,
with residuals `(6.6436e-8, 4.63224e-7)` against goals
`(0.0063485, 4.66294e-7)`. The next increment converged in 149 sweeps, then
failed the bounds guard: minimum `1.187073211580402e-6`, maximum
`1.016049756844138`, bounds tolerance `1e-8`. A repeated run reproduced these
values. A separate coarse experiment (`-rs 1 -rp 2`) also failed the bounds
guard early and was not used as evidence of the original stagnation.

A bounded physical interval extending past the reported stall was completed on
both one and four ranks:

```bash
# Working directory: /tmp/opencode/phase-stagnation
# Use N=1 or N=4. The one-rank run took about 359 seconds in this environment.
/usr/bin/time -f 'exit=%x elapsed_seconds=%e' timeout 400s mpiexec -n N \
  /home/miaodi/repo/mfemMechanics/build/release/bin/pPhaseField_shear \
  -m /home/miaodi/repo/mfemMechanics/data/crack_square2d_quad.msh \
  -rp 2 -il 1 -ni 1000 -tf 0.107 -disp 1.07e-5 -no-vis
```

The last two options preserve the imposed displacement/pseudo-time slope
`1e-4` and stop at 10.7% of the original physical loading. They also **tighten**
the example's dimensionally shared absolute tolerance from `4.66294e-7` to
`4.98934e-8`; these are not identical-tolerance replays of the full run.
Both runs reached their requested endpoint, passed the bounds checks, and
required 429 sweeps for their final increment. Final residuals were:

| Ranks | displacement residual | phase residual |
| --- | --- | --- |
| 1 | `2.62265e-8` | `4.90282e-8` |
| 4 | `2.64607e-8` | `4.90282e-8` |

The final goals were `(0.00572336, 4.98934e-8)` in both runs. This checks global
residual convergence, not equality of every solution component or reaction
curve. An initial one-rank attempt with a 120-second limit timed out and was
repeated with the larger bound above.

MPI still uses `ParMesh`, `ParFiniteElementSpace`, `ParGridFunction`, and
`ParBlockNonlinearForm`, with local true-DOF block vectors and distributed
Hypre operators/solvers. Neither the serial/distributed ownership model nor
collective control flow was changed. The MPI example remains gated by MFEM MPI;
no global gather or GPU path was introduced.

## Regression and remaining limitations

`ExactBlockSolvesDoNotGuaranteeCoupledConvergence` checks the analytical
near-stagnation residual and rollback with exact scalar diagonal solves.
`LargerSweepBudgetResolvesStrongCouplingWithoutRelaxingTolerance` checks
rejection at 12 sweeps, then convergence at the identical tolerance with a
larger budget, verifying both returned-state residuals and the analytical
solution. Existing tests cover current-state block reactivation and history
commit/rollback.

After the changes, the complete Debug and Release builds and CTest suites passed
(121/121 in each configuration), including serial, one-rank, and two-rank shear
smoke tests:

```bash
cmake --build build/debug --parallel 2
ctest --test-dir build/debug --output-on-failure
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
git diff --check
```

The full-load phase-bounds failure remains unresolved. The implemented
unconstrained consistent finite-element phase equation does not enforce nodal
box constraints. A bound-preserving discretization or constrained solve is a
separate mathematical change requiring its own residual/linearization and
verification. Neither increasing the iteration budget nor silently clamping the
phase is a justified fix for that limitation.

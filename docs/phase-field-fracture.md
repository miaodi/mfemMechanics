# Small-strain AT2 phase-field fracture

## Scope

`PhaseFieldElasticMaterial` and `plugin::PhaseFieldIntegrator` implement
small-strain isotropic elasticity coupled to an AT2 damage field. The unknown
blocks are displacement `u` and damage `phi`, in that order. There is no inertia,
viscosity, plasticity or explicit crack-face contact. Two-dimensional mechanics
is plane strain, with three-dimensional constitutive tensors.

The parallel shear example completes its full displacement range using a
staggered outer iteration with inner mechanical Newton solves. **The outer
iteration is not coupled Newton and does not generally converge quadratically:
crack propagation can require hundreds of sweeps.** Nodal phase overshoot remains a
discretization limitation assessed in postprocessing. Completion is not
an admissible, mesh-converged reproduction of the published benchmark.

## Conventions, parameters and units

```text
strain = sym(grad(u))
phi = 0: intact; phi = 1: fully damaged
c = 1 - phi: the paper's intactness variable
l_code = 2*l0_Borden
```

The shear inputs are in SI units: coordinates and displacement in m, E and
stress in Pa, positive energy and history H in J/m³, Gc in J/m², and l in m.
The continuation coordinate is dimensionless, not physical time. The integrated
2D mechanical residual/reaction has units N/m of out-of-plane thickness; the
phase residual has units N. Multiply a reaction by thickness in m to obtain N.

Default fracture parameters are Gc=2700, l=15e-6 and residual stiffness k=1e-9.
The paper sets k=0; the small positive default is an explicit numerical difference.
Require finite E>0, -1<nu<0.5, Gc>0, l>0 and 0<=k<1. Near-incompressibility can
be ill-conditioned; this is not a mixed incompressibility formulation.

Voigt ordering is `[xx, yy, zz, xy, yz, xz]`. Strain uses engineering shear
`[eps_xx,eps_yy,eps_zz,2 eps_xy,2 eps_yz,2 eps_xz]`; stress shears are unscaled.
The 6-by-6 tangent maps engineering strain increments to stress increments.
Plane strain sets `eps_zz=eps_yz=eps_xz=0` but retains out-of-plane stress.

## Constitutive law

Let `lambda=E*nu/((1+nu)*(1-2*nu))`, `mu=E/(2*(1+nu))`, `K=lambda+2*mu/3`,
and `t=trace(strain)`. Positive/negative parts mean `max(value,0)` and
`min(value,0)`. The stored elastic energy is

```text
psi = g(phi)*psi_positive + psi_negative
g   = (1-k)*(1-phi)^2 + k
g'  = -2*(1-k)*(1-phi)
g'' = 2*(1-k)
```

| Split | Positive energy | Negative energy |
| --- | --- | --- |
| Miehe spectral | `lambda/2*max(t,0)^2 + mu*strain_positive:strain_positive` | `lambda/2*min(t,0)^2 + mu*strain_negative:strain_negative` |
| Amor volumetric/deviatoric | `K/2*max(t,0)^2 + mu*dev(strain):dev(strain)` | `K/2*min(t,0)^2` |
| Isotropic | Full undamaged elastic energy | Zero |

The shear example uses the spectral split. The other splits remain available
as library models. Isotropic degradation reduces compression as well as tension;
neither split implements geometric nonpenetration.

These energies adapt Borden et al. [1], §2.2, equations (7), (14)–(16), and the
Miehe/Amor sources [2,3]. Stress is the derivative with respect to engineering
strain `e`: `s=g*s_positive+s_negative`; the material tangent is `C=ds/de`.
The stress derivative with respect to phase is `g'*s_positive`.

The spectral tangent uses an analytic matrix-function derivative rather than
derivatives of individual eigenvectors. For `strain=Q*diag(a)*Q^T`, define
`f(a)=max(a,0)` and divided differences L:

```text
L_ij = (f(a_i)-f(a_j))/(a_i-a_j)   for opposite signs
L_ij = 1                         for positive/same-nonnegative pairs
L_ij = 0                         for negative/same-nonpositive pairs
L_ij = 1/2                       when both values are zero
D strain_positive[D] = Q * (L elementwise (Q^T*D*Q)) * Q^T
```

The trace derivative similarly chooses 1, 0, or 1/2 at positive, negative, or
zero trace. These are centered generalized derivatives at nonsmooth switches;
repeated nonzero roots are handled without an eigenvalue-gap regularization.
Positive and negative contributions are weighted directly to retain very small
residual stiffness without cancellation. Each tangent column uses the correct
engineering-Voigt basis increment. `EvaluateResponse(true)` returns stress,
positive energy/stress, phase derivative, and an owning optional tangent;
`EvaluateResponse(false)` skips tangent computation.

## AT2 equation and history

The fracture density is

```text
Gc/2 * (phi^2/l + l*|grad(phi)|^2).
```

Substituting `c=1-phi` and `l=2*l0` in Borden's equation (6) gives this expression.
Thus the default l=15 µm matches the paper's l0=7.5 µm; halving the code's l
would change the model.

History is stored per quadrature point, initially zero:

```text
H_trial = max(H_committed, psi_positive(current strain))
-Gc*l*Laplacian(phi) + (Gc/l + 2*(1-k)*H_trial)*phi = 2*(1-k)*H_trial.
```

This adapts the history substitution in [1], equations (20)–(22). Natural phase
boundary conditions are `grad(phi) dot n=0`. At fixed displacement/history, the
AT2 phase equation is linear. The spectral mechanical equation is generally
nonlinear even at fixed phi, because its strain decomposition changes with u.

`BeginStep` initializes trial history from committed history. Every residual
or Jacobian evaluation recomputes it from the same committed state; rejected
Newton iterates do not accumulate maxima. Only a converged load step commits.
Failure restores unknowns and history, including nested-transaction rejection.
AMR/repartition after history develops requires a transfer not provided here;
the examples refine only before initialization. Restart serialization is not
implemented.

## Discrete residual and Jacobian

With `B*u_e=e`, `N^T*phi_e=phi`, physical shape-gradient matrix G, and quadrature
weight `w=ip.weight*det(dX/dxi)`, the implementation's contributions are

```text
Ru       += w * B^T*s
Rphi     += w * [g'*H*N + Gc*(l*G*G^T*phi_e + phi/l*N)]
Kuu      += w * B^T*C*B
Ku_phi   += w * (B^T*g'*s_positive)*N^T
Kphi_u   += w * g'*N*chi*s_positive^T*B
Kphi_phi += w * [Gc*l*G*G^T + (Gc/l + g''*H)*N*N^T]
```

Here `chi=1` when the current positive energy exceeds committed history and
zero otherwise, including equality. The cross blocks transpose on active
history; unloading generally gives a nonsymmetric full Jacobian. No geometric
stiffness is required in small strain. These discrete derivatives are derived
for this implementation from the stated laws.

Both assemblies use the same rule, default order `2*max(p_u,p_phi)+1`.
`SetIntRule` can supply a borrowed rule. Spectral splits are nonpolynomial, so
quadrature sensitivity remains part of numerical verification. Local element
displacement arrays are component-major regardless of the global ordering.

## Staggered solver and convergence

Despite its class name, `NewtonForPhaseField` is a nonlinear block
Gauss–Seidel solver for the coupled problem. Each load increment follows this
procedure:

1. Hold phi fixed and Newton-solve displacement using the current Kuu.
2. Solve the phase equation at the updated u and trial H.
3. Reevaluate both residuals at that common state. Repeat until both pass.
4. Commit history only after outer convergence; otherwise roll back the attempt.

The configured linear solver is reused for each block. The inner loop does not
create another history transaction or commit state. Both residuals must pass at
the same displacement/phase state; convergence of mechanics is never latched
across a phase update. There is no inner line search.

### Why hundreds of outer sweeps can be necessary

**`-ni` limits staggered sweeps, not Newton corrections. A 20-iteration Newton
rule of thumb does not apply to this outer loop.** Only the fixed-phase
mechanical subproblem uses Newton iteration. The phase subproblem is linear at
fixed displacement/history.

A full coupled Newton method would compute both corrections together:

```text
[ Kuu      Ku_phi   ] [ delta_u   ] = -[ Ru   ]
[ Kphi_u   Kphi_phi ] [ delta_phi ]    [ Rphi ]
```

The staggered method instead solves with Kuu and Kphi_phi in sequence. Although
the integrator assembles the cross derivatives, this solver does not use them
to predict the simultaneous response of both fields. Its local convergence is
generally linear when the fixed-point iteration is contractive, rather than
quadratic as for smooth, nonsingular Newton close to a solution.

During crack propagation, a phase update changes stiffness, displacement
redistributes, and the changed crack-driving energy activates further damage.
This strong feedback can make the staggered contraction very slow; the
residual may also grow before entering a contracting regime. As an illustration,
an error reduction factor of 0.98 per sweep takes about 570 sweeps to reduce
the error by 1e-5. This is an illustration, not a measured factor for the example.

**Retain the 1000-sweep default for the verified shear setup:** the propagation
run needed up to **515 outer sweeps**, while any mechanical subsolve needed at
most **4 Newton corrections**. A generous outer budget is necessary for this
method and configuration to finish without premature rejection. Exactly 1000
is neither a mathematical minimum nor a guarantee for other meshes or loads;
it provides headroom over the observed requirement.

Load cutbacks do not replace adequate coupling iterations. With `-ni 20` and
`-u-atol 1e2`, the example stalls at 11.531% of the load: each mechanical solve
reduces its residual from about 100 to 2.6e-8 N/m, but the phase update restores
an imbalance slightly above the 100 N/m outer goal. Repeated cutbacks reach
the minimum increment without resolving that imbalance. Raising the sweep
budget addresses this failure; lowering the minimum increment does not.

The large budget accommodates slow convergence rather than accelerating it.
A globalized coupled Newton or accelerated staggered method would be a separate
solver design, with history/spectral branch handling and convergence checks.

### Stopping criteria and load continuation

Outer goals are `max(rtol*reference, block_atol)`. The displacement reference is
its first nonzero residual in the attempted step. The phase reference is taken
after the first completed mechanical subsolve, with a first-subsequent-nonzero
fallback. An initial phase imbalance is still checked before any acceptance.

Each inner mechanical solve freezes its own initial norm and uses

```text
inner_goal = min(max(inner_rtol*inner_initial_norm, inner_atol), outer_u_goal).
```

An inner iteration-limit failure, iterative linear-solver failure, or nonfinite
correction/residual rejects the load attempt before updating phase. The inner
solve can take zero corrections when already converged. Both outer blocks must
still pass after the phase update; mechanical convergence is never latched.

The library retains `SetAbsTol()` as a shared fallback. `SetBlockAbsTol(u,phi)`
sets separate floors and `ClearBlockAbsTol()` restores that fallback.
`SetMechanicsNewton(iterations,rtol,atol)` configures the inner solve.

Parallel-example defaults, in its SI units:

| Control | Default |
| --- | --- |
| Outer relative tolerance, `-rtol` | 1e-5 (1e-4 for single precision) |
| Outer mechanics floor, `-u-atol` | 1e-2 N/m |
| Outer phase floor, `-phi-atol` | 1e-6 N |
| Maximum staggered outer sweeps, `-ni` | **1000** |
| Maximum inner Newton corrections, `-u-ni` | 30 |
| Inner relative tolerance, `-u-rtol` | 1e-8 (1e-5 for single precision) |
| Inner absolute tolerance, `-u-inner-atol` | 1e-3 N/m |
| Initial / maximum / minimum increment, `-dt` / `-dt-max` / `-dt-min` | 1e-6 / 1e-2 / 1e-14 |
| Final continuation coordinate, `-tf` | 1 |
| Final top displacement, `-disp` | 1e-4 m |

These are problem-specific tolerances. Separate floors prevent a tiny new load
increment from demanding another relative reduction of an already accepted
residual. The serial example has its own existing tolerances but uses the same
nested solver. Logs distinguish outer sweeps from inner mechanical iterations.

Failed attempts halve the increment. Successful increments grow by at most 1.2
up to `-dt-max`; the controller also uses the outer iteration budget, so changing
`-ni` can change the load sequence. `-steps` counts attempts including failures.
The driver checks that the requested final continuation coordinate was reached.

### Independent solver comparison

[PhaseFieldX 0.4.0, commit a9714c1](https://github.com/CastillonMiguel/phasefieldx/tree/a9714c122497f8829860efc68d776e4324d48eca)
was deployed with DOLFINx 0.11.0 and PETSc/MUMPS and run on a matching 4,096-quad
mesh. Its [history solver](https://github.com/CastillonMiguel/phasefieldx/blob/a9714c122497f8829860efc68d776e4324d48eca/src/phasefieldx/Element/Phase_Field_Fracture/solver/solver_history.py#L176-L406)
uses nonlinear displacement subsolves, separate inner tolerances, and projected
history. It provides an independent comparison for the subsolve structure and
physically scaled stopping criteria.

The upstream [shear example](https://github.com/CastillonMiguel/phasefieldx/blob/a9714c122497f8829860efc68d776e4324d48eca/examples/PhaseFieldFracture/plot_1712.py#L254-L335)
specifies two passes per increment. That policy completed the matching coarse
case through 0.0134 mm, with peak reaction 0.61610 kN/mm of thickness. It did
not require outer coupled convergence and exhibited nodal overshoot. A separate
convergence-controlled comparison exhausted 1000 passes at 0.0117 mm. Thus an
external example's completion is not proof that our coupled criteria should
pass in two sweeps. The reference's numerical methods differ; it is evidence
for solver design, not an independent certification of this formulation.

## Shear setup, ownership and execution

The MPI example uses the spectral material with E=210 GPa and nu=.3. The
provided mesh is a 1 mm square with a 0.5 mm notch ending at the centre and a
10 µm mouth opening. Borden [1], §4.1/Figures 5–7 uses a zero-width slit and
cubic C2 T-splines, so the geometry/discretization are not identical.

| Boundary attribute | MPI displacement condition |
| --- | --- |
| 11, bottom | ux=uy=0 |
| 12, top | ux ramps to `-disp`, uy=0 |
| 13, right; 14/15, left | uy=0; horizontal motion free |
| Notch faces | Traction-free |

The side restraints follow Figure 5(a), also independently implemented by the
PhaseFieldX example. No phase Dirichlet condition or initial diffuse crack is
imposed. The serial example still leaves the outer sides traction-free and is
therefore a different boundary-value problem.

The MPI defaults `-rs 3 -rp 2 -lr 0` give 4,096 Q1 quads with nominal
h=15.625 µm, a coarse solver demonstration. Explicitly selecting `-rp 4`
gives h≈3.906 µm, near the paper's smallest element size. Matching
h does not establish equal Q1/spline accuracy. Mesh, load-increment and quadrature
convergence are still needed.

MPI uses `ParMesh`, `ParFiniteElementSpace`, `ParGridFunction` and
`ParBlockNonlinearForm` with Hypre ParCSR gradient blocks. Each rank solves and
sums owned true DOFs; output fields are prolonged to local/shared DOFs. Reactions
are the unconstrained top-horizontal residual summed once across ranks. CSV
writing is rank-zero-only; norm, reaction and timing collectives run on all ranks.
There are no global solution gathers.

`-ls direct` requires MFEM MPI+MUMPS and uses separate distributed factorizations
for u and phi. The default `-ls gmres` uses separate GMRES/BoomerAMG solvers; `-uamg` selects
MFEM's systems/scalar/elasticity interpolation bundles. Matrices are rebound
on every update. `-il 1` reports linear residuals and timings. The target requires
MPI; MUMPS-dependent tests are gated separately. Serial uses UMFPack/SuiteSparse.

Forms own their integrators; coefficients, material and point storage are
borrowed and outlive the forms. `IntegrationPointStorage` binds the reference
finite element and exact rule object. Reset it after geometry/topology/rule
changes, transferring accepted history first. Materials and storage contain
mutable scratch and are not thread-safe; these Eigen-based kernels are host-only.

```bash
cmake --build build/release --target pPhaseField_shear --parallel 4
# From a disposable directory, with REPO set to the absolute repository path:
mpiexec -n 8 "$REPO/build/release/bin/pPhaseField_shear" \
  -m "$REPO/data/crack_square2d_quad.msh"
```

`-no-vis` disables both ParaView and CSV. `-od` controls the ParaView directory;
`p_phase_field_force.csv` is written and flushed in the working directory.

## Verification

### Full-load shear runs

The coarse setup above completes the full 0.1 mm displacement on one and eight
MPI ranks. Eight-rank Release tolerance comparisons give:

| GMRES controls | Accepted steps | Rejections | Maximum outer sweeps | Maximum inner corrections |
| --- | ---: | ---: | ---: | ---: |
| `-u-atol 1e2 -ni 1000` | 146 | 0 | 353 | 4 |
| `-u-atol 1e-2 -ni 1000` (defaults) | 146 | 0 | 515 | 4 |

The tighter run matched the direct-solver reference force CSV at printed
precision on the same 146 load samples. The looser run differed by up to
1,673 N/m, or 0.123% of the tighter run's peak reaction. Retain the tighter
tolerance and allow the outer iteration to converge rather than loosening the
goal. `-ni 20` explicitly overrides the working default; omit it or use
`-ni 1000`.

With default controls and an explicit mesh path, the one-rank run also completed
146 steps without rejection, with the same maximum sweep/correction counts.
Its force CSV matched the eight-rank run and the direct reference at printed
precision. Wall times were 286.41 s on eight ranks and 1523.10 s on one rank;
these timings are not a scaling study.

### Reproducibility and regression coverage

These runs used source `15aa715` plus the default-control changes in this
section, the existing Release preset build, GNU C++, MFEM version integer
40901 in double precision with MPI/MUMPS enabled and MFEM OpenMP disabled,
and the host-only assembly path. The GMRES settings were relative tolerance
1e-10, restart 50 and maximum 2000 iterations, with systems AMG for mechanics;
the mesh, material, loading and nonlinear settings are specified above.

Regression coverage in `tests/phase_field_test.cpp` checks engineering shear,
compression and split limits, zero strain, repeated roots, tiny residual
stiffness, centered material and four-block Jacobian differences, the homogeneous
AT2 balance, phase-aware postprocessing, independent tolerance floors, fixed-phase
inner Newton and transactional failure. `czm_history_test.cpp` covers
quadrature-history and nested-lifecycle behavior. One- and two-rank CTest cases
exercise direct/iterative assembly and solver rebinding.

Full Debug and Release builds and
`ctest --test-dir build/<configuration> --output-on-failure -j 4` each passed
all 179 enabled tests (two existing contact benchmark tests disabled).
Logs and force curves for the failure and comparisons are under
`/tmp/opencode/shear-tuning-20260910/` in the validation workspace.

### Bounds and interpretation limits

The history method guarantees nondecreasing committed H, not discrete bounds
or nodal monotonicity of phi. Consistent Q1 reaction–diffusion assembly need not
satisfy a discrete maximum principle; an accurate linear solve can overshoot.
There is no phase clipping or constitutive floor. The quadratic law extrapolated
outside [0,1] can regain stiffness and has no intended physical interpretation.

The MPI example does not use phase extrema to accept or reject a solve. It
checks finite displacement/phase coefficients, nonlinear convergence and the
final load target. Bounds must be assessed separately during mesh, quadrature
and load-increment verification; there is no bound-constrained evolution or
pre-commit admissibility retry. Nodal bounds alone also do not establish
pointwise bounds for higher-order elements. The serial example retains its
existing nodal-bound check. Do not use the unavailable inherited element energy
method for physical energy checks or globalization.

Bounds diagnostics on the direct reference recorded nodal violations at 91
accepted steps. The completed runs therefore establish algebraic convergence
and regression consistency, not physical admissibility. The 0.1 mm target is
beyond the paper's 0.0134 mm comparison range. Single-precision builds, full
mesh/increment/quadrature convergence, strict admissibility and published-curve
agreement have not been established.

## References

1. M. J. Borden, C. V. Verhoosel, M. A. Scott, T. J. R. Hughes and C. M. Landis,
   *A phase-field description of dynamic brittle fracture*, Computer Methods in
   Applied Mechanics and Engineering 217–220 (2012), 77–95.
   [DOI](https://doi.org/10.1016/j.cma.2012.01.008). Journal §2.2, equations
   (6)–(7), (14)–(22), and §4.1/Figures 5–7 were checked against the Zotero PDF.
2. C. Miehe, M. Hofacker and F. Welschinger, *A phase field model for
   rate-independent crack propagation: Robust algorithmic implementation based
   on operator splits*, CMAME 199 (2010), 2765–2778.
   [DOI](https://doi.org/10.1016/j.cma.2010.04.011). Spectral split/history in
   §3.1 and shear benchmark/loading in §5.2.
3. H. Amor, J.-J. Marigo and C. Maurini, *Regularized formulation of the
   variational brittle fracture with unilateral contact: Numerical experiments*,
   Journal of the Mechanics and Physics of Solids 57 (2009), 1209–1229.
   [DOI](https://doi.org/10.1016/j.jmps.2009.04.011), §4.3, equations (33)–(35).
4. M. Castillon, *PhaseFieldX*, version 0.4.0, source commit a9714c1. The pinned
   source links above identify the solver and shear driver used for comparison.

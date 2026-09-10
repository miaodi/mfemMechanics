# Small-strain AT2 phase-field fracture

## Commit status: more verification is needed

The current shear example is **not an identical, verified reproduction of the
Borden benchmark**, and no full fracture convergence or published reaction-curve
agreement is claimed. The comparison target is Borden et al., ICES Report 11-14
(2011), [author report](https://www.oden.utexas.edu/media/reports/2011/1114.pdf),
§4.1 and Figures 5–7; the convention conversion uses §2.2, equations (6)–(7).

- The material values `E=210 GPa`, `nu=0.3`, `Gc=2700 J/m²`, and `l=15 µm`
  match the report's `epsilon=7.5 µm` convention through `l=2 epsilon`.
  Agreement of these inputs alone does not establish benchmark equivalence.
- The supplied quadrilateral mesh has a finite triangular notch with a 10 µm
  opening, rather than the report's discrete slit. The investigated
  `-rs 3 -rp 2 -o 1` configuration has nominal `h=15.625 µm` and Q1 elements,
  versus the report's `h_min=3.906 µm` and cubic C2 T-splines. This investigated
  configuration is not the current MPI refinement default.
- The MPI example interprets the side rollers in the journal paper's Figure 5(a)
  as vertical restraints with horizontal motion free; the serial example still
  leaves those edges traction-free. The journal paper states in §2.2 that all
  calculations use `k=0`. Load stepping and reaction-force thickness normalization
  still require comparison. The code uses `k=1e-9` and reports
  2D reactions in N/m; a direct comparison to a plotted force needs an explicit
  thickness and unit conversion.

More verification is needed for these problem-definition differences,
mesh/order/quadrature and load-increment sensitivity, phase admissibility, and
the complete fracture history. No geometry or numerical-method change is made
as part of this commit preparation.

The dated verification records below and in the companion solver notes and
`phase-field-formulation.tex` are historical snapshots of intermediate dirty
trees, not proof that later source/default edits were tested. In particular,
the current MPI defaults are 15 nonlinear sweeps, double-precision relative
tolerance `1e-5`, maximum continuation increment `1e-2`, and `-rs 3 -rp 4`;
older descriptions of 12/25 sweeps, `1e-7`, or `1e-3` do not describe these
defaults. The MPI example now uses relative-only nonlinear convergence
(`abs_tol=0` for both blocks); historical checks below predate this change.
Fresh commit-time checks are reported separately from those records.

### Fresh commit-time checks, 2026-09-05

Using the existing workspace Debug/Release configurations, after the current
source/default edits:

```bash
cmake --build build/debug --parallel 4
ctest --test-dir build/debug --output-on-failure
cmake --build build/release --target pPhaseField_shear PhaseField_shear \
  phase_field_test eigen_decomp_test autodiff_test czm_history_test --parallel 4
ctest --test-dir build/release \
  -R 'PhaseField|EigenDecomp|autodiff|StressCoefficient' --output-on-failure
```

The full Debug suite passed **125/125**; the selected Release suite passed
**39/39**, including serial, one-rank, and two-rank smoke/repeated-solve cases.
Touched C++ passed `clang-format --dry-run --Werror`, and `git diff --check`
passed. These checks do not exercise the full default fracture history.
The full Release suite, single-precision build, LaTeX rebuild, and benchmark
comparison were not rerun for this commit.

## Scope and conventions

`PhaseFieldElasticMaterial` and `plugin::PhaseFieldIntegrator` implement
quasi-static isotropic brittle fracture with quadratic degradation and a
second-order AT2 crack functional. Displacement `u` and damage `phi` are the two
unknown blocks, in that order. `phi=0` is intact and `phi=1` is cracked. There is
no inertia, viscosity, plasticity, temperature evolution, or explicit crack-face
contact. The shear examples are plane strain; all constitutive tensors remain
three-dimensional with `eps_zz=eps_yz=eps_xz=0`. This is not the complete dynamic
Borden model.

The reference and current configurations coincide to small-strain accuracy:
`eps=sym(grad(u))`. The integrator constructs `F=I+grad(u)` only to use the
material interface; its B matrix and tangent are small-strain quantities. Do
not enable the inherited large-deformation flag for this formulation.
Coefficients, material, point storage, and custom quadrature rules are borrowed
and must outlive the integrator. Materials and storage have mutable scratch and
are not reentrant or thread-safe. Assembly is host-only.

The engineering-strain Voigt vector is
`e=[eps_xx,eps_yy,eps_zz,2 eps_xy,2 eps_yz,2 eps_xz]`; stress uses unscaled
shears in the same order. Thus `sigma:deps=s^T de` and `C=ds/de`. Tensor input
to the spectral decomposition already contains physical shear, so it must not be halved
again.

Use a consistent unit system. In the examples, lengths/displacements are metres,
E and stresses are Pa, energy density and history H are J/m³, and Gc is J/m².
The length `l` is in metres and residual stiffness `k` is dimensionless. The
continuation coordinate is dimensionless, not physical time. The integrated
2D horizontal reaction is force per unit out-of-plane thickness (N/m); multiply
by the physical thickness in metres for force in N.

Require finite `E>0`, `-1<nu<0.5`, `Gc>0`, `l>0`, and `0<=k<1`.
The defaults are `Gc=2700`, `l=15e-6`, `k=1e-9`. The near-incompressible limit
is ill-conditioned and is not a mixed incompressibility formulation; `k=0` can
produce singular displacement blocks. Finite phase values outside [0,1] are
permitted during nonlinear iterations; the library does not clamp them.

## Energies and stress

Write `lambda=E nu/((1+nu)(1-2nu))`, `mu=E/(2(1+nu))`,
`K=lambda+2mu/3`, `t=tr(eps)`, and `<x>+=max(x,0)`, `<x>-=min(x,0)`.
All splits recover `psi0=lambda t²/2+mu eps:eps` at zero damage.

| `StrainEnergySplit` | `psi+` | `psi-` |
| --- | --- | --- |
| `MieheSpectral` | `lambda <t>+²/2 + mu eps+:eps+` | `lambda <t>-²/2 + mu eps-:eps-` |
| `AmorVolumetricDeviatoric` | `K <t>+²/2 + mu dev(eps):dev(eps)` | `K <t>-²/2` |
| `Isotropic` | `psi0` | `0` |

Here `eps±=sum_i <eps_i>± n_i tensor n_i` and
`dev(eps)=eps-t I/3`. The spectral definitions adapt Miehe, Hofacker and
Welschinger (2010), §3.1.1–3.1.2, equations (13), (19), p. 2768.
The volumetric/deviatoric split adapts Amor, Marigo and Maurini (2009), §4.3,
equations (33)–(35), p. 1217. The isotropic option degrades compression too.
Neither split is a geometric nonpenetration constraint.

The stored energy is `psi=g(phi) psi+ + psi-`, with
`g=(1-k)(1-phi)²+k`, `g'=-2(1-k)(1-phi)`, `g''=2(1-k)`.
This normalized degradation follows Borden et al.'s author report, §2.2,
equation (7), rather than Miehe's unnormalized additive residual stiffness.
Consequently `g(0)=1` exactly. The AT2 fracture density is
`Gc/2 (phi²/l + l |grad(phi)|²)`. Substitution of `c=1-phi`, `l=2 epsilon`
in Borden's equation (6) gives this density. These are notation adaptations;
the discrete derivatives below are derived for this implementation.

Define `s+=d psi+/de`, `s-=d psi-/de`, `s=g s+ + s-`,
`C=g d s+/de + d s-/de`. The phase stress derivative is `g' s+`.
All material getters route through public `EvaluateResponse(computeTangent)`,
which obtains strain, elastic constants, and degradation once per call and
dispatches to one private analytic helper per split. Its owning `Response`
contains positive energy, positive stress, total stress, and `g' s+`;
`tangent` is an `std::optional<Eigen::Matrix6r>`, engaged exactly when requested.
`EvaluateResponse(false)` skips tangent assembly, rather than supplying a
misleading zero stiffness. There is no persistent response cache. Each residual
point uses one stress-only response and each Jacobian point one tangent response,
including both coupling blocks. History updates remain in the integrator with
the unchanged strict `psi+ > H_committed` active-derivative choice. Standalone
getters still reevaluate on each call; callers needing several quantities should
retain a local response. No autodiff is used by this material.
The isotropic and Amor formulas below are original elementary
differentiations of the stated energies, not additional cited formulas. For a
symmetric strain increment D:

```text
Isotropic:
  psi+   = lambda/2 t^2 + mu eps:eps
  sigma+ = lambda t I + 2 mu eps
  sigma- = 0
  C[D]   = g (lambda tr(D) I + 2 mu D)
Amor:
  psi+   = K/2 max(t,0)^2 + mu dev(eps):dev(eps)
  sigma+ = K max(t,0) I + 2 mu dev(eps)
  sigma- = K min(t,0) I
  C[D]   = K (g h(t) + 1-h(t)) tr(D) I + 2 mu g dev(D)
```

Here `h(t)=1` for positive t, `0` for negative t, and `1/2` at zero.
The zero-trace Amor tangent deliberately corrects the previous AD branch
selection, which assigned zero to both volumetric branch derivatives.
Complementary slopes `h` and `1-h` instead give the centered volumetric tangent
`K (g+1)/2` at zero trace and recover intact bulk stiffness when `g=1`, including
at zero strain and in pure shear. At damage greater than zero this is a
generalized derivative choice, not classical differentiability at the switch.
Every tangent column uses an engineering-Voigt basis increment (off-diagonal
tensor entries are 1/2 for a unit engineering shear). The material tangents
are symmetric energy derivatives on smooth branches, with symmetric centered
choices at the switches.

For the spectral branch, differentiating eigenvectors with autodiff gives
incorrect tangents at repeated eigenvalues. The implementation instead uses
the matrix-function derivative. For `eps=Q diag(a) Q^T` and symmetric increment
`D`, let `f(a)=max(a,0)` and

```text
D eps+[D] = Q (L elementwise-multiply (Q^T D Q)) Q^T
L_ij = (f(a_i)-f(a_j))/(a_i-a_j)                 if signs differ
L_ij = 1                                      if both nonnegative, not both zero
L_ij = 0                                      if both nonpositive, not both zero
L_ij = 1/2                                    if both zero
D sigma+[D] = lambda h(tr(eps)) tr(D) I + 2 mu D eps+[D]
D sigma[D] = lambda tr(D) I + 2 mu D + (g-1) D sigma+[D]
```

`h(t)` is 1 for positive t, 0 for negative t, and 1/2 at zero. This is the
repository's derivation: differentiate `Q f(diag(a)) Q^T` off repeated roots,
then take the same-sign divided-difference limit. No small eigenvalue-gap
regularization is needed. Each engineering-Voigt basis increment supplies one
column of C. The implementation weights positive and negative stresses/slopes
directly to avoid cancellation of very small residual stiffness in pure tension.
The centered generalized slope at zero is a stated branch choice,
not a claim of classical differentiability there. Repeated nonzero principal
strains are smooth and checked by directional finite differences.

## Residual and consistent four-block Jacobian

At a quadrature point, let `B u_e=e`, `N^T phi_e=phi`, and
`G^T phi_e=grad(phi)`, where rows of G are physical shape gradients. Set
`H=max(H_n,psi+(e))` from the last committed value H_n. With
`w=ip.weight*det(dX/dxi)`, element contributions are

```text
Ru   += w B^T s
Rphi += w [g' H N + Gc (l G G^T phi_e + phi/l N)]
Kuu      += w B^T C B
Ku_phi   += w (B^T g' s+) N^T
Kphi_u   += w g' N chi (s+)^T B
Kphi_phi += w [Gc l G G^T + (Gc/l + g'' H) N N^T]
```

`chi=1` when `psi+>H_n`, otherwise 0, including equality. This explicitly
chooses the inactive derivative at the history switch. On the active branch
the cross blocks transpose each other; on unloading `Kphi_u=0` while `Ku_phi`
generally remains nonzero. The full Jacobian is therefore generally
nonsymmetric. There is no geometric stiffness in this small-strain formulation.
External mechanical loads subtract their weak-form contribution from Ru.
Natural phase boundary conditions are `grad(phi) dot n=0` in the examples.

Both assemblies use the same rule: by default order
`2 max(p_u,p_phi)+1`; `SetIntRule` supplies a borrowed override. Spectral splits
are nonpolynomial, so convergence studies must also increase quadrature order.
Point storage binds the displacement finite element and the exact rule object;
reset it if either changes. Local element displacement arrays are
component-major regardless of global MFEM ordering.

The history substitution is not direct constrained minimization of damage.
Only the quadrature history H is guaranteed nondecreasing across accepted
steps. Arbitrary FE meshes and higher-order spaces do not have a discrete
maximum principle, and pointwise `phi_(n+1)>=phi_n` or `0<=phi<=1` is not imposed.
The examples check coefficient bounds after acceptance and fail if violated;
this diagnostic does not implement a bound-constrained solve or cut back an
inadmissible phase state. For higher-order elements coefficient bounds alone
do not establish bounds everywhere inside the element. No physical energy
check should use the inherited unavailable element-energy method.

## State lifecycle and solver

History initializes to zero. `BeginStep` copies committed history to trial;
every residual/Jacobian evaluation recomputes trial history from H_n and the
current fields, so rejected Newton iterates do not accumulate maxima. Accepted
steps commit the final residual's trial history. Failure rolls history and
unknowns back; nested rejection prevents the enclosing transaction committing.
Geometry/refinement changes after history develops require an admissible state
transfer that is not implemented. The examples refine only before initialization
and do not perform AMR during fracture. Restart serialization is not provided.

Despite its name, `NewtonForPhaseField` performs block Gauss-Seidel updates:
solve the displacement diagonal block, reevaluate both residuals, solve the
phase diagonal block, and repeat. It does not use the cross blocks for a
monolithic Newton solve and does not perform an energy line search. Both block
residual norms must satisfy their own `max(rel_tol*reference_norm,abs_tol)` from
iteration zero. Each block independently fixes its reference at its first
nonzero residual norm in the current solve, including evaluations between block
updates. An initial nonzero reference is retained; an initially zero block uses
only `abs_tol` until coupling activates it, then establishes its relative
reference even when `abs_tol=0`. A supplied RHS is subtracted on every evaluation.
After either field update both residuals are recomputed at the same current state. A block
within tolerance can skip its current solve, but its convergence is never
latched: an update to the other field can reactivate it. Acceptance requires
both current residuals to pass simultaneously. The shared
absolute tolerance acts on blocks with different units and may need problem-
specific scaling. The MPI example uses 15 sweeps by default, relative tolerance
`1e-5` in double precision (`1e-4` in single), and zero absolute tolerance for
both blocks. Its references reset for each attempted increment. This avoids
skipping phase updates because of a displacement-derived absolute threshold,
but very small reference residuals can demand accuracy below floating-point
resolution. Full-fracture convergence with this setting remains to be verified.
The serial example retains its existing absolute-tolerance policy.

Adaptive continuation restores the accepted solution before prescribing each
trial boundary value. Failed solves halve the increment; successful solves
grow it by at most 1.2 up to `-dt-max`. CLI inputs require finite
`0<dt-min<=dt<=dt-max` and `dt-min<dt-max`. `-steps` counts attempts, including
rejections. Exhaustion or failure to reach the final coordinate is reported.

## Serial and MPI shear examples

Both examples set `E=210e9 Pa`, `nu=0.3`, use the spectral split, and require a
2D planar mesh with domain attribute 1 and bottom/top attributes 11/12. Bottom
and top vertical displacement are zero; bottom horizontal displacement is zero
and top horizontal displacement ramps to `-disp`. The MPI example additionally
requires right attribute 13 and left attributes 14/15 and imposes zero vertical
displacement there, leaving horizontal motion free, following the side rollers
in Borden et al. (2012), §4.1, Figure 5(a), p. 85
([journal article](https://doi.org/10.1016/j.cma.2012.01.008)). It combines
component-specific owned true-DOF lists for both residual elimination and trial
boundary values. The serial example still leaves the outer sides traction-free,
so the two examples currently solve different boundary-value problems. Crack
faces are traction-free in both. There is no prescribed phase boundary or initial diffuse
crack; the supplied mesh defines the notch.

The serial default mesh is `data/crack_square2d.msh`; MPI defaults to
`data/crack_square2d_quad.msh`, with different refinement defaults. Use the same
explicit mesh and final refinement when comparing ranks. `-lr` is optional
problem-specific initial refinement, with different serial/MPI selection rules.
The MPI polynomial crack-path selector is a legacy heuristic in metres, not a
predicted path or portable refinement criterion for arbitrary meshes.

| Serial | MPI |
| --- | --- |
| `Mesh`, `FiniteElementSpace`, `GridFunction` | `ParMesh`, `ParFiniteElementSpace`, `ParGridFunction` |
| `BlockNonlinearForm` | `ParBlockNonlinearForm` with Hypre ParCSR gradient blocks |
| UMFPack diagonal solves (SuiteSparse required) | Distributed MUMPS diagonal solves by default (`-ls direct`, MFEM MPI + MUMPS required); `-ls gmres` retains systems/scalar BoomerAMG, 2000 iterations, restart 50, relative tolerance `1e-10` (`1e-5` single) |
| True-DOF vector prolonged for output, including nonconforming meshes | Each rank owns its true DOFs; `SetFromTrueDofs` populates shared/local output fields |
| Continuous H1 stress projection | Discontinuous stress projection |

In `pPhaseField_shear`, `-il 1` (or `--info-level 1`) prints a rank-zero
summary after every linear solve, labeled by block. Direct solves report the true
relative residual and maximum-rank analysis/factorization and solve times, with
no Krylov iteration or preconditioned-residual labels. For `-ls gmres` the summary reports total iterations across restarts, the true
relative residual `||b-Ax||/||b||`, the preconditioned final/initial residual
ratio reported by MFEM, convergence status, and maximum-rank setup/solve times. Summaries include solves in
rejected continuation attempts. The default `-il 0` disables these summaries
and their extra matrix multiply/global norms. A zero denominator reports zero
for a zero numerator, otherwise infinity. This option controls linear summaries;
existing nonlinear progress output is unchanged.

With `-ls gmres`, the default displacement AMG uses `SetSystemsOptions(2)` with the existing
`Ordering::byVDIM` space. `-uamg scalar` restores the previous scalar AMG
settings; `-uamg elasticity` and `-uamg elasticity-no-refine` enable MFEM's
rigid-body-mode interpolation with/without interpolation refinement.
`NewtonForPhaseField::SetBlockSolvers(displacement, phase)` borrows separate
linear solvers; `SetSolver(shared)` retains the original shared-solver behavior.
AMG is rebuilt for each new tangent, so this change does not reuse stale
hierarchies or alter nonlinear acceptance/tolerances. See
[Eight-rank AMG tuning](phase-field-amg-tuning.md) for measurements and limits.
See [direct solver selection](phase-field-direct-solver.md) for capability gates,
factorization semantics, verification, and the unchanged staggered iteration.

The forms own their integrators and are destroyed before the borrowed material
and point storage. Reaction output assembles an unconstrained internal residual
and sums top horizontal true-DOF entries. MPI reduces the owned entries once;
global DOF counts, residual norms and diagnostics are collective on all ranks,
with CSV output only on rank zero. There are no global solution gathers.
ParaView writes use MFEM's parallel data collection. `-no-vis` disables both
ParaView and reaction CSV; `-od` changes the ParaView directory, while CSV is
written in the working directory.

From the repository root, these matched small runs use explicit inputs and
write no output files:

```bash
cmake --preset debug
cmake --build --preset debug --parallel 4
build/debug/bin/PhaseField_shear -m "$PWD/data/crack_square2d_quad.msh" \
  -r 0 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1 -no-vis
mpiexec -n 1 build/debug/bin/pPhaseField_shear -m "$PWD/data/crack_square2d_quad.msh" \
  -rs 0 -rp 0 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1 -no-vis
mpiexec -n 2 build/debug/bin/pPhaseField_shear -m "$PWD/data/crack_square2d_quad.msh" \
  -rs 0 -rp 0 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1 -no-vis
ctest --preset debug
```

For force curves, run the absolute executable and mesh paths from a disposable
working directory with `-vis`. Record source revision and local diff, preset,
MFEM precision/version/capabilities, mesh, order/refinement, all CLI parameters,
solver tolerances, ranks/threads and accepted increments. The default parameters
are inspired by Borden's shear problem (author report §4.1); these tiny smoke
runs do not validate crack propagation, mesh convergence, or a published curve.

## Verification and API migration

`phase_field_test` checks engineering shear, compression/split behavior,
constitutive tangents, repeated principal strains, active and inactive history
Jacobians, initial phase residual, nonzero RHS, phase-aware stress output, and
all von Mises shear terms. `czm_history_test` contains `PhaseFieldHistory` and
`PhaseFieldIntegrator` lifecycle tests for deterministic trial evaluation and
nested commit/rollback. CTest also includes serial, one-rank and two-rank smoke
runs. Run full Debug and Release suites for solver/numerical changes; smoke
success alone establishes neither benchmark agreement nor performance.

Replace the old `StrainEnergyType` enum with `StrainEnergySplit` and choose
`MieheSpectral`, `AmorVolumetricDeviatoric`, or `Isotropic` explicitly. The old
`Borden` label is not an energy split and has no compatibility alias. Removed
fallbacks that returned zero energy are unsupported rather than physical zero.
Supply `PhaseFieldFractureParameters` to change Gc, l and k. Material instances
are noncopyable/nonmovable; E/nu coefficients remain borrowed. Stress output
must call `StressCoefficient::SetPhaseField` as well as `SetDisplacement`.

The imported custom 3-by-3 eigensolver has been removed. The unused-by-material
`util::StrainSplit` helper retains its positive/negative tensor return contract
using `Eigen::SelfAdjointEigenSolver`, but now requires finite symmetric input
and built-in floating-point scalars. Autodiff instantiations are explicitly
unsupported; no AD-to-real conversion is performed. The material continues to
use Eigen with the analytic spectral derivative above. `eigen_decomp_test`
checks reconstruction, orthogonality, repeated eigenvalues, scaling and physical
shear splitting; the former custom-solver AD comparison is no longer verification
coverage. Unrelated autodiff tests and the autodiff dependency remain.

## References and verification provenance

- C. Miehe, M. Hofacker, F. Welschinger, *A phase field model for rate-independent
  crack propagation: Robust algorithmic implementation based on operator splits*,
  Computer Methods in Applied Mechanics and Engineering 199 (2010), 2765–2778.
  [DOI](https://doi.org/10.1016/j.cma.2010.04.011).
  Equations (13), (19), (23), pp. 2768–2769 were checked in the
  [mirrored primary article](https://www.scribd.com/document/585804616/miehe2010).
  The repository uses a different iterative block order and normalized g.
- H. Amor, J.-J. Marigo, C. Maurini, *Regularized formulation of the variational
  brittle fracture with unilateral contact: Numerical experiments*, Journal of
  the Mechanics and Physics of Solids 57 (2009), 1209–1229.
  [DOI](https://doi.org/10.1016/j.jmps.2009.04.011).
  §4.3, equations (33)–(35), p. 1217 were checked against the locally retained
  primary-article extraction `/tmp/opencode/amor-2009.txt`; the DOI endpoint
  was unavailable during this review. The split is adapted, not its full solver.
- M. J. Borden, C. V. Verhoosel, M. A. Scott, T. J. R. Hughes, C. M. Landis,
  *A phase-field description of dynamic brittle fracture*, ICES Report 11-14
  (2011), [author report](https://www.oden.utexas.edu/media/reports/2011/1114.pdf).
  §2.2, equations (6)–(12), pp. 5–6, and §4.1 were checked in the author report
  and retained extraction. The journal version is CMAME 217–220 (2012), 77–95,
  [DOI](https://doi.org/10.1016/j.cma.2012.01.008); equation/page references here
  identify the report, not an unverified journal pagination.

## Review verification, 2026-09-05

Verified against base revision `528c91c09c4d0d6a95a1396eeb345a14678ce5d0`
plus this uncommitted phase-field diff, using the repository Debug/Release
presets, GCC 15.2.0 and MFEM 4.9.1 in double precision. MFEM has MPI and
SuiteSparse enabled, OpenMP and CUDA disabled; assembly used the CPU.
Release flags were `-O3 -DNDEBUG`. No random data or solver option files were
used. Single-precision compilation was not performed.

Commands completed:

```bash
cmake --preset debug
cmake --preset release
cmake --build --preset debug --parallel 4
cmake --build --preset release --parallel 4
build/debug/bin/phase_field_test
build/release/bin/phase_field_test
ctest --test-dir build/debug --output-on-failure
ctest --test-dir build/release --output-on-failure
git diff --check
```

Both full suites passed **106/106**, including serial/one-rank/two-rank shear
smokes; both focused executables passed **11/11**. MPI initially could not open
PMIx sockets in the sandbox; the full suites passed outside that restriction.
SemismoothContact has no current test registration; its unrelated stash was
preserved. Existing rigid-obstacle contact tests remain part of the base suite.

Matched Release output runs used the quadrilateral mesh, order 1, no refinement,
`-disp 1e-9 -tf 1 -dt 0.25 -dt-max 0.25 -steps 4 -vis`, with serial `-r 0`
and MPI `-rs 0 -rp 0`. All four increments converged and all three CSVs agreed
at their printed precision, ending at reaction **64.3949 N/m**. Artifacts and
logs are under `/tmp/phase-review/{serial,one,two}`. A Debug serial output run
with the same mesh, `-r 0 -lr 1 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1
-vis` also converged, exercising nonconforming output reconstruction.

A deliberately larger matched run on the unrefined mesh with `-disp 1e-7`
and four increments converged algebraically on the first increment but failed
the phase coefficient bound check in serial and MPI. This is an observed
limitation of the unconstrained history formulation on that discretization;
no bounds were clamped and no tolerance was relaxed. The default full fracture
run, a published reaction-curve comparison, mesh/quadrature convergence, and
full-fracture/scaling benchmarks remain unperformed. Subsequent bounded AMG
performance measurements are recorded in [Eight-rank AMG tuning](phase-field-amg-tuning.md).

### Local response reuse measurement, 2026-09-05

This measurement compares the dirty analytic implementation present at the start
of the response-reuse edit (same base revision above, **not** the committed AD
implementation) with the public response and split helpers. Existing
`build/release` configuration was reused: GCC 15.2.0, `-O3 -DNDEBUG`, MFEM 4.9.1
double precision, CPU i9-10900KF. No reconfiguration was needed.

The temporary harness `/tmp/opencode/phase-response-bench.cpp` uses the element
assembly wrappers in `tests/phase_field_test.cpp`: one unit-square Q1 quad,
order-3 quadrature (four points), spectral plane strain, E=10, nu=0.25,
Gc=2.5, l=0.3, k=0.01 in consistent arbitrary units. The fixed fields are
`u=(0.018x+0.004y, 0.004x-0.007y)`, `phi=0.2+0.03x+0.02y`. Committed history
is zero; the active trial is repeatedly evaluated inside one begin/rollback
transaction. Each batch assembles 100,000 residual/Jacobian pairs, including
wrapper allocation and block copying, and accumulates `||R||2+||J||F`.
One warmup batch is excluded, followed by seven timed batches. No solver,
nonlinear iterations, MPI collectives, visualization, or random data are involved.

`bash /tmp/opencode/build-phase-bench.sh before` was run **before source edits**;
the same script with `after` rebuilt the harness after the Release libraries.
Both binaries were run from `/tmp/opencode` with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Initial median seconds per batch
were **1.21573 before / 0.87769 after** (27.8% less time). A subsequent
before/after pair under `/usr/bin/time -f 'peak_RSS_kB=%M'` gave medians
**1.18381 / 0.89050 seconds** (24.8% less time); peak process RSS was
27,336 / 26,892 KiB. CPU affinity/frequency were not pinned. All batch checksums
agreed at 17 printed digits, ending at **13309887.659665646** including warmup.
This supports a local assembly improvement only, not a full-solve or scaling claim.

Coefficient-counter assertions now verify one E and one nu evaluation per point
in both residual and Jacobian, for all three splits with active and inactive
history (previously residual: two; Jacobian: four active / three inactive).
The existing four-block directional-difference test now covers all three splits;
a focused response test checks getters, snapshot reuse, and optional-tangent
presence. Full rebuilds with `cmake --build build/{debug,release} --parallel 4`
and `ctest --test-dir build/{debug,release} --output-on-failure` each passed
**119/119**, including serial/one-rank/two-rank shear smokes. The focused Debug
`ctest --test-dir build/debug -R PhaseField --output-on-failure` passed 22/22.
Single-precision builds and a before/after full nonlinear solve were not run.

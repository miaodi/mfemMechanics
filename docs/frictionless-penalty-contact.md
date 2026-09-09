# Frictionless contact with a rigid obstacle

## Status and design goal

This feature implements unilateral, frictionless contact between a deformable
body and a smooth analytic rigid obstacle. Two enforcement methods are
available: a displacement-only penalty integrator and a monolithic semismooth
mixed operator with boundary Lagrange multipliers. Neither method stores hidden
contact history.

The obstacle is analytic rather than a second finite-element mesh. Contact
search, two-body contact, friction, adhesion, cohesive bonding, self-contact,
and MPI assembly are outside the implemented scope. The mixed method is an
experimental serial capability: its equations, consistent generalized
Jacobian, jump stabilization, and small benchmark are tested, but broad element
pair stability and mesh-convergence studies have not been completed.

**API notation update (2026-09-08):** the mixed operator takes the paper's
compliance `gamma = 1/kappa` and dimensionless jump factor `delta`; the
displacement-only penalty integrator still takes stiffness `kappa`. This is a
reciprocal parameterization of the same mixed model, not new physics. Numerical
results and test counts below are retained historical snapshots, not reruns of
the migrated API. See the [boundary multiplier note](boundary-multiplier-space.md)
for basis-aware equations and source correspondence, and the
[formulation source](semismooth-rigid-contact-formulation.tex) for the derivation
and implementation map.

**Boundary-space update (2026-09-09):** the mixed operator now borrows a
`BoundaryMultiplierSpace`, with default P0 or configurable scalar,
value-mapped discontinuous L2 multipliers. See the
[boundary multiplier note](boundary-multiplier-space.md) for ownership,
basis-aware assembly, P1/Q1 regression constructions, and geometry restrictions.
P0-specific discussion and historical results below remain scoped to P0;
no tests were rerun for this documentation update.

## Kinematics and sign convention

Let \(\mathbf X\) be a point on the initial material contact boundary
\(\Gamma_{c0}\), and let

$$
\mathbf x(\mathbf X) = \mathbf X + \mathbf u_h(\mathbf X)
$$

be its current spatial position. The rigid obstacle supplies a signed-distance
function \(g(\mathbf x)\) with

$$
g > 0 \quad \text{separated}, \qquad
g = 0 \quad \text{touching}, \qquad
g < 0 \quad \text{penetrating}.
$$

Its derivatives are

$$
\mathbf n = \nabla_{\mathbf x}g,
\qquad
\mathbf H = \nabla_{\mathbf x}^2 g.
$$

The admissible normal \(\mathbf n\) points toward increasing gap. A valid
custom obstacle must provide one coherent signed-distance evaluation, not an
arbitrary level-set value combined with a separately normalized normal. In the
smooth contact tube,

$$
\lVert\mathbf n\rVert = 1,
\qquad
\mathbf H = \mathbf H^T,
\qquad
\mathbf H\mathbf n = \mathbf 0.
$$

The signed distance need only be smooth where it is evaluated. Points with a
non-unique closest projection, such as the center of a circular or spherical
obstacle, are inadmissible.

The implementation provides:

- an arbitrarily oriented plane, with
  \(g=(\mathbf x-\mathbf x_0)\mathbin\cdot\mathbf n\) and \(\mathbf H=\mathbf0\);
- the exterior of a circle in 2D or sphere in 3D, with
  \(g=\lVert\mathbf x-\mathbf c\rVert-R\) and
  \(\mathbf H=(\mathbf I-\mathbf n\otimes\mathbf n)/\lVert\mathbf x-\mathbf c\rVert\);
- a runtime interface for other smooth signed-distance functions, including
  nonhorizontal curves.

## Penalty potential, residual, and tangent

The normal penalty stiffness \(\kappa>0\) has units of stress per length. The
contact potential per unit initial boundary measure is

$$
\psi_c(g) = \frac{\kappa}{2}\langle-g\rangle_+^2
          = \frac{\kappa}{2}\min(g,0)^2.
$$

The associated compressive pressure magnitude is

$$
p_n = \kappa\langle-g\rangle_+.
$$

This pressure never produces adhesion or tangential traction. With shape
functions collected in the vector interpolation operator \(\mathbf N\), the
contact residual, written as the gradient of the potential, is

$$
\mathbf r_c
  = \int_{\Gamma_{c0}}
      \kappa\min(g,0)\,\mathbf N^T\mathbf n\,dA_0.
$$

The physical contact force on the body is \(-\mathbf r_c\). Away from the
activation point, the algorithmically consistent tangent is

$$
\mathbf K_c
  = \int_{\Gamma_{c0}}
      \kappa\mathbf N^T
      \left(\mathbf n\otimes\mathbf n + g\mathbf H\right)
      \mathbf N\,dA_0,
  \qquad g<0,
$$

and zero for \(g>0\). The \(g\mathbf H\) term is required for a curved
obstacle. Omitting it gives an inconsistent normal-variation linearization.
Its tangential contribution can be negative during penetration, so a general
curved-contact tangent must not be assumed positive definite.

At \(g=0\), the potential is continuously differentiable but has no classical
second derivative. This implementation chooses the active, one-sided tangent
while retaining a zero residual. That convention gives normal support to a
load-controlled body initially touching a plane. No hidden gap tolerance is
used.

Both residual and tangent use the initial material-boundary measure
\(dA_0\). A current-boundary measure would introduce additional follower-load
terms and is not implemented.

## Semismooth mixed formulation

The mixed method solves displacement or a larger primal field vector together
with a normal-pressure multiplier. Its contact conditions are

$$
g \geq 0, \qquad \lambda \geq 0, \qquad \lambda g = 0,
$$

where \(\lambda\) is positive in compression and has stress units. Use the
paper's compliance \(\gamma>0\), with units of length per stress, and define

$$
q = \lambda_h - g/\gamma,
\qquad
p = [q]_+ = \max(0,q),
\qquad
\chi =
\begin{cases}
1, & q \geq 0,\\
0, & q < 0.
\end{cases}
$$

The active branch is deliberately selected at \(q=0\). Pointwise,
\(p=\lambda_h\) is equivalent to the three complementarity conditions above.
With a boundary P0 multiplier, the unstabilized multiplier equation enforces
the P0 projection of that equality rather than pointwise equality when the gap
varies along a face. Jump stabilization modifies that projected equation by
the consistent \(\mathbf S\boldsymbol\lambda\) term below. The contact part of
the discrete saddle functional is

$$
\Pi_c(\mathbf u_h,\lambda_h)
= \frac{\gamma}{2}\int_{\Gamma_{c0}}(p^2-\lambda_h^2)\,dA_0
- \frac{1}{2}s(\lambda_h,\lambda_h).
$$

Here \(\mathbf u\) is vector displacement, \(\mathbf v\) its variation/test,
and \(\mu\) the multiplier test. In the paper's scalar contact expression the
sign map is \(u_{\mathrm{source}}=-g\),
\(\lambda_{\mathrm{source}}=-\lambda\); its test variations are
\(v_{\mathrm{source}}=-\mathbf n\cdot\mathbf v\) and
\(\mu_{\mathrm{source}}=-\mu\). This maps the contact algebra, not the scalar
bulk PDE to vector elasticity. With the reference-boundary inner product
\(\langle a,b\rangle=\int_{\Gamma_{c0}}ab\,dA_0\), first variation gives

$$
r_u(\mathbf v)=-\langle p,\mathbf n\cdot\mathbf v\rangle,
\qquad
r_\lambda(\mu)=\gamma\langle\mu,p-\lambda_h\rangle-s(\lambda_h,\mu).
$$

Let \(\mathbf N\) interpolate displacement and let \(M\) interpolate the
multiplier. The coefficient residuals are

$$
\mathbf r_u
= -\int_{\Gamma_{c0}}p\,\mathbf N^T\mathbf n\,dA_0,
$$

$$
\mathbf r_\lambda
= \gamma\int_{\Gamma_{c0}}M^T(p-\lambda_h)\,dA_0
- \mathbf S\boldsymbol\lambda.
$$

At \(\lambda_h=0\) and \(g<0\), \(p=-g/\gamma\), so the displacement
residual is exactly the penalty residual with \(\kappa=1/\gamma\). On a smooth
active-set branch, the generalized Jacobian blocks are

$$
\mathbf K_{uu}
= \int_{\Gamma_{c0}}
  \mathbf N^T\left(\frac{\chi}{\gamma}\,\mathbf n\otimes\mathbf n-p\mathbf H\right)
  \mathbf N\,dA_0,
$$

$$
\mathbf K_{u\lambda}
= -\int_{\Gamma_{c0}}\chi\,\mathbf N^T\mathbf n M\,dA_0,
\qquad
\mathbf K_{\lambda u}
= -\int_{\Gamma_{c0}}\chi\,M^T\mathbf n^T\mathbf N\,dA_0,
$$

$$
\mathbf K_{\lambda\lambda}
= \gamma\int_{\Gamma_{c0}}(\chi-1)M^TM\,dA_0
- \mathbf S.
$$

These are the exact derivatives of the stated discrete residual away from an
active-set transition and the selected active one-sided derivatives at a
transition. In particular, the curved-obstacle term is \(-p\mathbf H\), not
the penalty-only term \(\kappa g\mathbf H\).

### Boundary P0 multiplier and stabilization

The **default** multiplier is discontinuous piecewise constant on an
`mfem::SubMesh::CreateFromBoundary()` mesh. There is exactly one scalar
multiplier DOF per selected parent boundary face in this P0 specialization.
MFEM requires those boundary attributes to form one connected subset.
`BoundaryMultiplierSpace` owns this mesh and its configurable L2 collection
and space; higher-degree basis products generalize the same residual and
Jacobian formulas above (where \(M\) is a row evaluation operator).

Discontinuous multipliers do not generally satisfy the required inf-sup
condition without stabilization. The implementation therefore includes the
negative jump term

$$
s(\lambda_h,\mu_h)
= \sum_{F\in\mathcal F_c}
  \delta\gamma h_F
  \int_F[\lambda_h][\mu_h] \, ds,
$$

where \(\mathcal F_c\) contains interior faces of the contact submesh,
\(h_F\) is the average \(d_c\)-th root of the measure of the two adjacent
\(d_c\)-dimensional contact elements, and
\(\delta\geq0\) is dimensionless. The default is \(\delta=1\). Setting it to
zero is permitted for controlled stability experiments, not recommended as a
general discretization. For a 2D body, an interior contact-submesh face is a
point and its zero-dimensional measure is one. In 3D, the interface length is
quadrature-integrated with an order derived from both adjacent geometric
transformations and deliberately oversampled because a curved transformation's
metric weight need not be polynomial.

This term adapts Burman, Hansbo, and Larson's discontinuous-multiplier jump
stabilization: Section 2, p. 4, equation (2.7), in the
[2016 v1 PDF](https://arxiv.org/pdf/1609.03326v1) associated with Zotero
`ZAV78EWP`. Its stabilized **Formulation 1**, p. 4, (2.10), is the
implemented contact algebra after the sign map above. The
multiplier-in-primal alternative (2.4)/(2.5) and stabilized Formulation 2,
(2.11)/(2.13), are not this implementation. In the separately verified 2016 v1
HTML text, the jump and Formulation 1 saddle equations are numbered (9) and
(12). These are source-specific locators, not a claim of final-journal
numbering equivalence; see reference 3 and the paper-to-code guide.

### Monolithic block operator and scaling

Let \(\mathbf y\) contain all primal true DOFs and let a user-supplied operator
\(\mathbf E\) extract displacement, \(\mathbf u=\mathbf E\mathbf y\). The
assembled nonlinear system is

$$
\begin{bmatrix}
\mathbf R_y\\
\mathbf R_\lambda
\end{bmatrix}
=
\begin{bmatrix}
\mathbf F_{\mathrm{primal}}(\mathbf y)+\mathbf E^T\mathbf r_u\\
\mathbf r_\lambda
\end{bmatrix}
= \mathbf0,
$$

with Jacobian

$$
\begin{bmatrix}
\mathbf J_{\mathrm{primal}}+\mathbf E^T\mathbf K_{uu}\mathbf E
  & \mathbf E^T\mathbf K_{u\lambda}\\
\mathbf K_{\lambda u}\mathbf E & \mathbf K_{\lambda\lambda}
\end{bmatrix}.
$$

`SemismoothRigidContactOperator` returns this as one `mfem::BlockOperator` and
exposes the wrapped primal form through `CompositeNonlinearOperator`, so
material-point `BeginStep`/`CommitStep`/`RollbackStep` callbacks still reach
stateful primal integrators. The extraction operator avoids assuming that
displacement is block zero and permits a `BlockNonlinearForm` or another
composite primal operator.

The constructor takes `gamma` in the former `penalty` argument position and
`delta` in the former final `multiplierJumpStabilization` position. `gamma`
must be positive and finite, with `1/gamma` representable as a positive finite
`mfem::real_t`; `delta` must be finite and nonnegative and defaults to 1.
The corresponding implementation members are `mGamma` and `mDelta`.
The intervening primal and multiplier residual row scales remain positive
finite scalars, each defaulting to 1.
The operator returns
\(\operatorname{diag}(s_u,s_\lambda)[\mathbf R_y,\mathbf R_\lambda]^T\)
and the correspondingly row-scaled Jacobian. Scaling does not change the
solution. Unequal scales make the represented Jacobian nonsymmetric even when
the unscaled saddle Hessian is symmetric, so the benchmark uses GMRES rather
than CG or MINRES. The 2D benchmark makes both residual blocks dimensionless
with

$$
F_0 = E U_{\max}W/H,
\qquad
G_0 = U_{\max}W,
\qquad
s_u=1/F_0,
\qquad
s_\lambda=1/G_0.
$$

Here the scalar \(E\) is Young's modulus, \(W\) and \(H\) are block width and
height, and \(U_{\max}\) is the maximum prescribed displacement. The extraction
operator \(\mathbf E\) and obstacle Hessian \(\mathbf H\) are distinct from
these scalar quantities. Different applications must choose reference force
and integrated-gap scales consistent with their own units and geometry.

## Discrete evaluation and ownership

At each boundary-face quadrature point, the penalty integrator:

1. maps mesh-face rule coordinates to boundary-element coordinates using the
   inverse orientation and sets the boundary transformation's integration point;
2. evaluates the boundary finite-element trace shape functions;
3. maps that point through the boundary transformation to obtain \(\mathbf X\);
4. interpolates displacement and forms \(\mathbf x=\mathbf X+\mathbf u_h\);
5. evaluates and validates \(g\), \(\mathbf n\), and \(\mathbf H\);
6. applies the residual or tangent formula with
   `ip.weight * transformation.Weight()` exactly once.

The semismooth mixed operator likewise uses `GetBE()`,
`GetBdrElementVDofs()`, and `GetBdrElementTransformation()` to evaluate the H1
trace and assemble all three displacement-containing contact blocks. Contact
depends only on the boundary value of displacement, not its volume gradient;
therefore trace-zero volume DOFs are omitted from the contact sparsity pattern
and local dense matrices, rather than stored as explicit zero couplings.
Signed DOF indices still pass through MFEM's extraction and assembly routines.

The mixed operator retains the **volume-based P0 rule selection** below and
accounts for the configured multiplier degree.
Rule points remain in mesh-face coordinates, including user-supplied rules.
It applies the inverse boundary-to-face orientation to each point before
evaluating the boundary FE and geometry. Thus even a nonsymmetric custom rule
samples the same physical points with the same weights; simply reusing its
coordinates on an independently oriented boundary element would not suffice.
The adjacent volume transformation is used only for rule selection, not
displacement interpolation or quadrature-point geometry evaluation.

For penalty contact and the mixed P0 default, integration order follows MFEM's
boundary-face heuristic,
`Elem1->OrderW() + 2 * element.GetOrder()` with one additional order for a
`Pk` element. The mixed operator uses the maximum of displacement and
multiplier orders in the doubled term. It evaluates both fields at the same
original parent boundary-element point; `BoundaryMultiplierSpace` verifies
the identity reference map from that element to its submesh element.
A finite rule is not exact for a general signed distance or for a
face crossed by the active-set boundary. Set the penalty integrator's `IntRule`
or call the mixed operator's `SetIntegrationRule()` and check quadrature
sensitivity for such problems; the same rule must be used for the residual and
tangent.

The implementations support full-dimensional H1 displacement fields in 2D and
3D under MFEM legacy assembly. Element-local vector values use MFEM's
component-major layout, independently of the finite-element space's global
`byNODES` or `byVDIM` ordering. Embedded manifolds, ND/RT fields, partial
assembly, matrix-free assembly, and device kernels are unsupported.
Applications using the penalty integrator must leave `mfem::NonlinearForm` at
its default `AssemblyLevel::LEGACY`; its boundary-element callbacks do not
provide a partial-assembly or matrix-free implementation.

The mixed operator is serial and additionally requires
`GetVSize() == GetTrueVSize()` and boundary traces without an MFEM
`DofTransformation`. It explicitly rejects parallel finite-element spaces,
nonconforming meshes, and every MFEM backend other than the unconfigured or
default `Backend::CPU` path, including alternate CPU backends. NURBS parent
meshes and curved `SubMesh` parents are unsupported; parent geometry nodes must
be uniform-order Gauss--Lobatto H1 for MFEM's coefficient-copy submesh transfer.
Discontinuous geometry is rejected because its transfer can duplicate parent
DOFs. These geometry restrictions do not apply to the multiplier basis.
The parent mesh, geometry, topology, spaces, and contact attributes
must not change during its lifetime. Reconstruct the boundary object and its
borrowers after such changes. Keep a custom rule alive and fixed during a solve;
`SetIntegrationRule` does not change cached jump quadrature.

### Boundary-trace verification and assembly timing

`SemismoothContact.HighOrderBoundaryTraceMatchesVolumeOracle` compares residuals
and generalized-Jacobian actions against independent volume-shape evaluation
on curved quadratic geometry with order-4 H1 triangles, quadrilaterals,
tetrahedra, and hexahedra, for both global vector layouts. It exercises active
and inactive faces, the sphere Hessian, default and nonsymmetric custom rules,
and centered directional differences. It checks both coupling actions outside
each face's trace and verifies that interior trace-zero displacement changes
produce no contact residual change or Jacobian action. These are numerical
support checks, not assertions about stored zero entries in the private CSR
blocks hidden behind MFEM product operators.

An opt-in, assembly-only timing test is available (disabled in normal CTest):

```bash
cmake --build build/release --target contact_test --parallel 4
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build/release/bin/contact_test --gtest_also_run_disabled_tests \
  --gtest_filter=SemismoothContact.DISABLED_TraceAssemblyBenchmark
```

Historical measurement on 2026-09-05, comparing the restored semismooth
implementation on `afad1f5` (retained stash
`wip: semismooth mixed contact before review`) with
the boundary-trace update, using the same test and existing Release build:
GCC 15.2.0, `-O3 -DNDEBUG`, Intel i9-10900KF, serial CPU, MFEM 4.9.1 double
precision (`3b6b6f295f569eee066fec48037003297d505955-dirty`). The installed
MFEM package was `../mfem/install/release/lib/cmake/mfem`; MPI, SuiteSparse,
MUMPS, and PETSc were enabled, while MFEM OpenMP and CUDA were disabled.
No reconfiguration was performed for this experiment.

The workload is a unit cube with 3-by-3-by-3 hexes, order-4 H1 displacement,
`byVDIM`, all 54 boundary faces selected, no essential constraints, zero
displacement, constant multiplier 100, sphere center (-2,-2,-2), radius 6,
the then-current penalty argument 17 (equivalent to compliance \(\gamma=1/17\)),
and unit row scales and \(\delta=1\). Units are arbitrary but consistent.
There is no primal integrator, nonlinear solve, random input,
or output file. Each sample averages 20 residual/Jacobian assembly pairs after
20 warmup pairs; setup is excluded. CPU affinity was not pinned.

| Implementation | Five samples (ms/pair) | Median (ms/pair) |
| --- | --- | --- |
| Adjacent volume DOFs | 126.330, 121.256, 122.338, 121.094, 120.051 | 121.256 |
| Boundary trace DOFs | 7.112, 7.030, 7.038, 6.995, 7.007 | 7.030 |

The measured median ratio is 17.2 for this isolated workload; both runs printed
residual norm 29.7788. This is not an end-to-end solver speedup or a scaling
claim, and peak memory was not measured. The volume-oracle regression provides
the tighter numerical comparison. In that snapshot, full Debug and Release
CTest each passed all 141 enabled tests, including the semismooth benchmark
and refinement checks.
Only the opt-in timing test was disabled there and was run separately above.

The boundary-trace update deliberately leaves library quadrature unchanged.
The example-level resolution policy described below addresses the coarse-circle
regression without changing that library contract.

### Penalty boundary-assembly measurement

Historical measurement (exact date not recorded here), on the dirty working
tree based on `afad1f5a2bf7a2a1a19132725a3047b91d3b0e3a`, using the existing
Release build:
GCC 15.2.0, `-O3 -DNDEBUG`, Intel i9-10900KF, MFEM 4.9.1 (reported git string
`heads/master-0-g3b6b6f295f569eee066fec48037003297d505955-dirty`), double precision,
default CPU backend. MFEM has MPI, SuiteSparse, MUMPS, PETSc and Tribol enabled;
OpenMP, CUDA and SLEPc disabled. This is serial assembly without a solver or
threaded assembly; affinity was not pinned. The existing cache uses
`MFEM_DIR=/home/miaodi/repo/mfem/install/release/lib/cmake/mfem`; it was not
reconfigured for this measurement.

```bash
cmake --build build/release --target contact_test --parallel 4
build/release/bin/contact_test \
  --gtest_filter='PenaltyContact.DISABLED_BoundaryAssemblyBenchmark' \
  --gtest_also_run_disabled_tests
```

The historical isolated penalty workload was a unit cube with 3-by-3-by-3 hexes, order-4 H1
displacement, `byVDIM`, all 54 boundary faces, no essential constraints, zero
displacement, sphere center (-2,-2,-2), radius 6 and penalty 17. Units are
arbitrary but consistent. Each route had 20 warmup residual/Jacobian pairs;
five samples of 20 pairs alternated legacy then boundary assembly in the same
executable. Setup and reporting were excluded. No random input was used.
The command above now runs a boundary-only benchmark with the same warmup and
sample counts; the removed face implementation is no longer timed or callable.

| Registration | Five samples (ms/pair) | Median | Stored Jacobian entries |
| --- | --- | --- | --- |
| `AddBdrFaceIntegrator` | 127.648, 127.995, 127.838, 127.766, 127.406 | 127.766 | 3,391,650 |
| `AddBoundaryIntegrator` | 6.83040, 6.86865, 6.86331, 6.88018, 6.78674 | 6.86331 | 279,954 |

The median ratio is 18.6 for this isolated assembly workload, not an
end-to-end solver speedup or scaling claim. Both residual norms were 6.63483;
the historical benchmark also checked full residual agreement. Peak memory was not
measured. The current `BoundaryAssemblyMatchesVolumeShapeOracle` regression
checks residual/Jacobian actions and energy against independent volume-shape
quadrature, as well as energy and residual
directional derivatives, CSR trace support, essential true-DOF handling, both
global orderings, curved order-4 triangles/quads/tetrahedra/hexes, and default
and nonsymmetric custom rules. At the time of the historical comparison, full
Debug and Release CTest each passed all 147 enabled tests. For the boundary-only
API migration, full Debug CTest passed all 147 enabled tests, and the Release
contact selection passed all 32 enabled tests (including seven example checks).
The two opt-in timing tests remain disabled in CTest; the boundary-only penalty
timing test was also run explicitly in Release as above. Contact MPI, single
precision, nonconforming meshes, wedges and pyramids were not tested in this migration.

### Lifetimes and essential constraints

For penalty contact, use `NonlinearForm::AddBoundaryIntegrator`. Its
`AssembleElementVector/Grad` callbacks receive the boundary FE and boundary
DOFs: the local tangent is genuinely trace-sized, with no padding to adjacent
volume DOFs and no stored contact entries involving interior DOFs. The circle
example uses this route. MFEM handles global ordering, restriction to true
DOFs, and essential row/column elimination; set the form's essential true DOFs
as for any other nonlinear form.

`AddBdrFaceIntegrator` registration is no longer supported by the penalty
integrator; its face callbacks have been removed. Direct callers must use
`GetBE()` and `GetBdrElementTransformation()` with boundary-local displacement
DOFs and the `AssembleElementVector/Grad` callbacks. This boundary route requires standard
value-mapped scalar H1 traces replicated by spatial dimension and a mesh-owned
boundary transformation, with equal trace/adjacent-volume polynomial order.
It is not a domain integrator or an adapter for arbitrary custom volume bases.
Default quadrature retains the adjacent volume transformation's `OrderW()`
and the triangle/tetrahedron volume `Pk` increment (not a triangular wedge
trace increment). Both default and custom rules retain mesh-face coordinates:
points are inverse-orientation mapped to boundary coordinates, preserving
weights. Thus existing custom face rules, including nonsymmetric ones, sample
the same physical points. The example's energy and gap diagnostics retain
their existing sampling; independent `GapSampling` is unchanged.

The integrator stores a non-owning reference to the obstacle. The obstacle must
outlive the nonlinear form because MFEM owns integrators passed to
`AddBoundaryIntegrator`, while the integrator continues to borrow the obstacle.
A custom integration rule and boundary marker are likewise owned by the caller
and must remain alive as required by MFEM. The integrator reuses mutable
assembly scratch and is not reentrant; threaded assembly requires one instance
per thread.

The mixed operator owns assembled contact blocks but borrows the
`BoundaryMultiplierSpace` that owns its submesh, collection, and FE space.
The boundary object in turn borrows the parent mesh. The operator also borrows
the primal operator, extraction operator, displacement space, obstacle, and
optional integration rule. All borrowed dependencies must outlive their
borrowers. The multiplier solution stays in the caller's monolithic vector.
Both the boundary object and contact operator prohibit copy/move; contact
rejects temporary extraction operators and obstacles. Mutable assembly and
operator scratch is not reentrant; remaining lifetimes are the caller's
responsibility. Public contact mesh/space/multiplier accessors are unchanged.

The displacement essential true-DOF list must describe the same constraints as
the primal operator after displacement extraction. Exact list equality is
checked for a direct `mfem::NonlinearForm` with identity extraction; a generic
extraction or block primal operator must maintain the contract itself.
Prescribed displacement values remain in the gap evaluation, while contact
residual rows and Jacobian variations at those DOFs are removed. Every selected
contact face must retain at least one free displacement trace DOF. A fully
assembly-sampled active face must additionally retain meaningful coupling of
the constant multiplier mode to free normal displacement variation after
essential masking. `mElementConstantNormalCoupling` evaluates that mode
directly, without assuming coefficients are nodal values. Failure is rejected,
but this restrictive local guard is not a full-rank/inf-sup test or a necessary
condition for every globally stabilized system to be nonsingular.

Like MFEM nonlinear forms generally, the block Jacobian returned by
`GetGradient()` is valid only until the next gradient evaluation. Its primal
block also borrows the reference returned by the wrapped primal operator's
`GetGradient()`: do not call that method directly while retaining the mixed
Jacobian. Refresh the mixed Jacobian through
`SemismoothRigidContactOperator::GetGradient()` instead.

For example, a horizontal obstacle at \(y=0\) is registered on selected
boundary attributes as follows:

```cpp
mfem::Vector point(2);
mfem::Vector normal(2);
point = 0.0;
normal = 0.0;
normal(1) = 1.0;

plugin::RigidPlaneObstacle obstacle(point, normal);
mfem::NonlinearForm residual(&displacement_space);
residual.AddBoundaryIntegrator(
    new plugin::FrictionlessPenaltyContactIntegrator(obstacle, penalty),
    contact_boundary_marker);
residual.SetEssentialTrueDofs(essential_displacement_true_dofs);
```

**Alternatively**, use the mixed method with a **contact-free** primal form.
Do not wrap the penalty residual above: the mixed operator already adds its
own contact residual and doing both would double count contact. The obstacle
and displacement space can be shared, but the forms below are distinct:

```cpp
mfem::NonlinearForm primal(&displacement_space);
// Add elasticity and other non-contact primal integrators to primal here.
primal.SetEssentialTrueDofs(essential_displacement_true_dofs);
mfem::Array<int> contact_attributes(1);
contact_attributes[0] = contact_attribute;
mfem::IdentityOperator extract_displacement(displacement_space.GetTrueVSize());
plugin::BoundaryMultiplierSpace multipliers(
    *displacement_space.GetMesh(), contact_attributes); // default P0

// Convert an existing stress/length penalty value to the paper's compliance.
// gamma must be positive and finite, with a positive finite representable reciprocal.
const mfem::real_t gamma = mfem::real_t{1.0} / penalty; // length/stress
const mfem::real_t delta = 1.0; // dimensionless multiplier-jump factor

plugin::SemismoothRigidContactOperator contact(
    primal,
    extract_displacement,
    displacement_space,
    obstacle,
    multipliers,
    essential_displacement_true_dofs,
    gamma,
    primal_residual_scale,
    multiplier_residual_scale,
    delta);

mfem::BlockVector unknown(contact.GetBlockOffsets());
unknown = 0.0;
```

The argument after `obstacle` is now the borrowed boundary object rather than
attributes. Scalar argument positions and row-scale defaults are unchanged,
but the former stiffness argument must be **inverted**, not merely renamed.
The circle example likewise passes `gamma = 1/penalty` to the mixed constructor.
Its historical penalty-factor CLI flags and output labels are not renamed by
this API change.

The penalty integrator implements `GetElementEnergy()` using the same
rule and gap as its residual and tangent, so its contribution to
`NonlinearForm::GetEnergy()` is available. This does not make unavailable energy
methods in other registered integrators usable. The mixed operator does not
implement `GetEnergy()`. Residual-based Newton methods remain usable for both
penalty and mixed contact.

## Penalty selection

A useful mesh-dependent starting value is

$$
\kappa = \alpha\frac{E_{\mathrm{eff}}}{h_c},
$$

where \(h_c\) is a representative contact-face size and \(\alpha\) is a
dimensionless factor. This is a numerical enforcement or augmentation
parameter, not a material property. Increasing \(\alpha\) reduces
penalty-method penetration and changes the conditioning and active-set scaling
of both formulations. The benchmark therefore reports penetration, contact
pressure, contact resultant, and nonlinear convergence, and is intended to be
repeated for several values such as \(\alpha=1,10,100\). The historical
`-gamma/--penalty-factor` option sets this dimensionless \(\alpha\), **not**
the paper's compliance \(\gamma\). The displacement-only penalty stiffness
and its selection rule are unchanged; only the prose symbol for the CLI factor
is disambiguated. In mixed mode the example converts the selected stiffness to
\(\gamma=1/\kappa=h_c/(\alpha E_{\mathrm{eff}})\); its jump coefficient is
\(\delta\gamma h_F\). Historical emitted penalty labels retain their meaning.

## Verification

Focused regression tests cover:

- plane and circle/sphere signed-distance values and derivatives;
- inactive contact and the active-at-zero tangent convention;
- residual sign, reference-face scaling, and zero tangential response for a
  penetrated plane;
- a curved-obstacle Jacobian against a centered directional finite difference,
  including tangential perturbations that detect a missing \(g\mathbf H\) term;
- equivalent assembled resultants for `mfem::Ordering::byVDIM` and
  `mfem::Ordering::byNODES`;
- one boundary P0 multiplier per selected parent face;
- mixed active, inactive, and active-at-equality branches in 2D, and the active
  plane residual and directional Jacobian in 3D;
- the complete row-scaled mixed Jacobian against a centered directional
  difference for a curved obstacle and a primal vector containing an unrelated
  extra field;
- elimination of essential displacement variations from all contact Jacobian
  blocks, agreement with a direct primal form's essential constraints, and
  rejection of a fully active face without a free normal displacement
  variation;
- the magnitude, sign, and conservation of the P0 multiplier jump term,
  including geometry-aware interface integration on a curved 3D contact
  surface; and
- lifecycle forwarding from a composite nonlinear operator to a nested
  state-aware primal form.

The [boundary multiplier regression map](boundary-multiplier-space.md#regression-map-and-verification-limits)
lists the added P1/Q1 assembly constructions separately; the historical pass
counts below are not evidence that these new tests have run. For a nonconstant
basis, `MinimumMultiplier`/`MaximumMultiplier` sample the interpolated field at
assembly quadrature points, not coefficient or exact function extrema.
Neither positivity nor nonpenetration between samples is guaranteed.

The standalone 2D benchmark drives the complete bottom boundary of a
rectangular elastic block upward while constraining horizontal motion on the
left boundary. The top boundary contacts an off-center rigid circle. The
default circle has center \((0.65,1.26)\), radius \(0.25\), and initial vertical
clearance \(0.01\) above a unit-square block; the bottom displacement ramps to
\(0.02\). The benchmark checks initial separation, prescribed-displacement
enforcement, nonlinear convergence, active contact, a downward contact
resultant, and localization within the circle's horizontal projection. The
circle geometry and displacement are configurable from the command line. The
executable uses penalty contact by default; `--semismooth` or `-sm` selects the
monolithic displacement--P0-multiplier system. The mixed mode additionally
checks multiplier admissibility, the pointwise contact-map residual
\(\gamma\lvert[\lambda_h-g/\gamma]_+-\lambda_h\rvert\), the scaled mixed
residual, and the multiplier DOF count. The contact-map diagnostic need not
vanish at a stabilized discrete solution because the multiplier equation also
contains the jump term.
The example deliberately retains default P0, its one-DOF-per-face check, and
its inactive-P0 diagonal multiplier preconditioner; these are not generic
higher-degree solver choices.

The `-r/--refine-level` option applies uniform mesh refinement before the
finite-element spaces are created. The benchmark updates \(h_c\), and therefore
\(\kappa=\alpha E/h_c\) and the mixed compliance
\(\gamma=h_c/(\alpha E)\), after every refinement. Each mixed run requires
exactly one multiplier DOF per selected parent contact face and rejects
penetration or a contact-map residual larger than the complete prescribed displacement. The
paired refinement CTest runs the fixed 2-by-2 and 4-by-4 cases and requires both
quantities to decrease. This is a mesh-consistency regression, not an estimated
convergence rate or a general bound for arbitrary refinements. Its ParaView
output registers the planar displacement as \((u_x,u_y,0)\) so vector filters
work consistently.
The penalty mode writes a closed polyline approximation of the analytic rigid
circle as a separate obstacle dataset. The mixed mode writes separate body,
boundary-multiplier, and obstacle datasets. `Warp By Vector` moves only the
deformable body dataset and leaves the obstacle fixed.

### Circle quadrature resolution and independent gap sampling

The example (both enforcement modes) explicitly sets one fixed segment Gauss
rule using `-cq/--contact-integration-order`, default **127** (64 points per
face). It passes the MFEM-owned, lifetime-stable rule to `SetIntegrationRule()`
or `SetIntRule()` before solving, so residual, Jacobian, and integral diagnostics
use exactly the same rule. The example requires an order at least twice the H1
order, but that polynomial condition alone is **not sufficient** for contact:
the circle distance is nonpolynomial and the positive-part contact map has a
kink at each active-set boundary. Neither the library default nor any fixed
order guarantees finding an arbitrarily narrow patch.

The default is a benchmark-specific, measured resolution choice. With
`--semismooth -nx 2 -ny 2 -o 2 -steps 4 -no-output`, the old heuristic selects
three points per face. All six miss the patch: the contact-free rigid upward
translation penetrates by 0.01 at x=0.65, yet its quadrature penetration is zero
and its multiplier vanishes. The new default detects contact and solves the
mixed problem. To reproduce the old underresolution explicitly, add `-cq 4`;
this still fails verification and now reports independently sampled penetration
of 0.01 and a resolution warning instead of only the misleading quadrature zero.

`-gs/--gap-sample-intervals`, default **200 per face**, controls a separate
endpoint-inclusive uniform diagnostic grid. It evaluates displacement using the
adjacent volume FE and face-to-volume mapping, independently of the mixed
operator's boundary-trace evaluation. Its sampled minimum gap and maximum
penetration are printed separately; sampled penetration also enters the mixed
benchmark's existing prescribed-displacement bound. These samples neither
modify the equations nor clip gaps. They are **not continuous collision
detection** or an upper bound on penetration between samples. Increase both
quadrature and diagnostic resolution, and refine the mesh, when changing circle
size, displacement, or approximation order. A small algebraic residual alone
does not establish geometric resolution.

Historical Release sensitivity for the command above on 2026-09-05, in the same
MFEM double-precision CPU configuration recorded in the boundary-trace section
(existing build, no dependency/configuration changes):

| `-cq` | Vertical resultant | Independent sampled penetration | Mixed residual norm |
| --- | --- | --- | --- |
| 31 | -4.35057831 | 0.00338965437 | 6.55e-15 |
| 63 | -4.27346490 | 0.00352765296 | 1.25e-12 |
| 127 (default) | -4.33710090 | 0.00343573648 | 1.94e-11 |
| 255 | -4.32848474 | 0.00344825454 | 9.87e-14 |
| 511 | -4.33045720 | 0.00344599618 | 1.05e-13 |

The 127-to-255 changes are about 0.20% in vertical force and 0.36% in sampled
penetration. This is sensitivity evidence, not an exact integration claim or a
convergence rate; active-set quadrature errors need not decrease monotonically.
The P0 multiplier and stabilization still allow nonzero pointwise penetration.
The coarse quadratic CTests exercise the default and order 255 full solves.
`SemismoothContact.CoarseQuadraticCircleNeedsResolvedQuadrature` additionally
reproduces the six-point miss, checks independent sampling against the exact
rigid-motion minimum, compares resolved force integrals to a continuous analytic
integral (1% relative tolerance), and checks the mixed directional Jacobian at
both rules and both global vector layouts. The existing curved order-4 volume
oracle continues to cover nonsymmetric-rule orientation and geometric mapping.
The independent sampler also has a curved-geometry, nonuniform-displacement
test whose analytic minimum lies at a face endpoint. The underresolution CTest
requires exit code 2, the resolution warning, and independently measured 0.01
penetration for `-cq 4`.

In the 2026-09-05 quadrature-fix snapshot on dirty revision `afad1f5`, the
existing Debug and Release builds each passed all **146 enabled CTests**,
including the unchanged paired mesh-refinement check; only the opt-in assembly
timing test was disabled.
Verification commands were:

```bash
cmake --build build/debug --parallel 4
ctest --test-dir build/debug -R 'Contact' --output-on-failure
ctest --test-dir build/debug --output-on-failure
cmake --build build/release --parallel 4
ctest --test-dir build/release --output-on-failure
```

Both configurations also passed default penalty and default mixed runs with
`-no-output`, and the coarse command above with `-o 4`. Increasing the quadratic
case's diagnostic grid from `-gs 200` to `-gs 800` left the reported sampled
penetration unchanged to the eight printed decimal places in scientific
notation. Single-precision and alternate MFEM capability configurations were
not run for this fix.

## References and provenance

1. Wriggers, P. (2006), *Computational Contact Mechanics*, 2nd edition,
   Springer, Berlin, Heidelberg.
   [doi:10.1007/978-3-540-32609-0](https://doi.org/10.1007/978-3-540-32609-0).
   Section 2.1.3 supplies the penalty regularization used here; the notation is
   adapted to this repository's positive-gap convention and a rigid analytic
   obstacle.

2. Alart, P. and Curnier, A. (1991), "A mixed formulation for frictional
   contact problems prone to Newton like solution methods," *Computer Methods
   in Applied Mechanics and Engineering*, 92(3), 353--375.
   [doi:10.1016/0045-7825(91)90022-X](https://doi.org/10.1016/0045-7825(91)90022-X).
   Historical provenance for the mixed penalty-duality contact map and its
   Newton-like solution strategy, attributed secondarily by reference 3,
   2016 v1 HTML Section 2, equation (3). The Alart--Curnier full text was not
   verified in the recorded review; no frictional formulation is claimed
   implemented here.

3. Burman, E., Hansbo, P., and Larson, M. G. (2019), "Augmented Lagrangian
   finite element methods for contact problems," *ESAIM: Mathematical
   Modelling and Numerical Analysis*, 53(1), 173--195.
   [doi:10.1051/m2an/2018047](https://doi.org/10.1051/m2an/2018047).
   This is the previously verified journal metadata, not the locator for the
   source equations used here. The primary-source locators are:
   - **2016 v1 HTML:** [arXiv:1609.03326v1](https://arxiv.org/html/1609.03326v1),
     dated 12 September 2016, Section 2, (3)--(5) for the contact relation,
     functional, and mixed pair; (9) for jumps; (12) for stabilized Formulation 1.
     Section 6's introduction specifies linear primal/constant multiplier
     experiments, and Section 6.3 gives the scalar Signorini example.
   - **2016 v1 PDF associated with Zotero `ZAV78EWP`:**
     [arXiv:1609.03326v1](https://arxiv.org/pdf/1609.03326v1),
     12 September 2016, 26 pages; Section 2,
     (2.1) relation, (2.2) functional, (2.3) positive-part mixed pair, (2.7)
     jumps, and (2.10) stabilized Formulation 1. The preceding source reading
     verified PDF pp. 1--4: p. 3 defines continuous H1 degree k and
     discontinuous L2 degree l=k-1; p. 4 contains (2.7) and (2.10).
     This documentation edit uses that supplied verification, not a new PDF
     reading or a claim of final-journal numbering. Do not substitute the
     alternative (2.4)/(2.5) or
     Formulation 2, (2.11)/(2.13), for the implemented (2.10).

The reference-boundary residuals and complete curved-obstacle tangents above are
derived in this document by first and second variation of the two stated
potentials. The rigid-obstacle Hessian terms, positive multiplier sign, active
generalized derivative at equality, residual row scaling, and exact API
validation rules are repository adaptations or implementation choices rather
than formulas attributed verbatim to the references.

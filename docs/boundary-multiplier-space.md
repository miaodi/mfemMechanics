# Boundary multiplier space

## Scope

`plugin::BoundaryMultiplierSpace` owns a selected reference-boundary submesh,
its finite-element collection, and its scalar finite-element space. It stores
**no multiplier solution**. `SemismoothRigidContactOperator` borrows this typed
space and assembles contact with its local basis, rather than assuming one
coefficient per face. The null-collection default remains discontinuous P0;
configurable discontinuous P1/Q1 has focused regression constructions.

The accepted collection is an `mfem::L2_FECollection` of nonnegative order,
with scalar, `VALUE`-mapped elements compatible with the boundary dimension
and geometries. Higher orders and MFEM's positive/Bernstein multiplier basis
use the same basis-aware assembly. Acceptance is not a stability, inf-sup,
quadrature-convergence, or accuracy guarantee for an arbitrary element pair.
Continuous multipliers, NURBS/IGA, vector and integral-mapped collections are
not supported by this abstraction.

Source/API: [`fem/BoundaryMultiplierSpace.h`](../fem/BoundaryMultiplierSpace.h),
[`fem/BoundaryMultiplierSpace.cpp`](../fem/BoundaryMultiplierSpace.cpp), and
[`fem/Contact.h`](../fem/Contact.h)/[`Contact.cpp`](../fem/Contact.cpp).
The [contact feature guide](frictionless-penalty-contact.md) supplies obstacle
and solver conventions; its examples remain P0-specific where stated. The
[formulation source](semismooth-rigid-contact-formulation.tex) gives the mixed
derivation and implementation map, with its stated discretization scope.

## Construction, access, and lifetime

Given an initialized serial H1 vector `displacement_space`, a **contact-free**
`primal` operator, a compatible fixed linear extraction `extract_displacement`,
and a long-lived `obstacle`, the default call is:

```cpp
#include "Contact.h"
#include <memory>
#include <utility>

// contact_attributes lists attribute IDs, not a boundary marker array.
plugin::BoundaryMultiplierSpace multipliers(
    *displacement_space.GetMesh(), contact_attributes); // nullptr -> P0

plugin::SemismoothRigidContactOperator contact(
    primal, extract_displacement, displacement_space, obstacle, multipliers,
    essential_displacement_true_dofs, gamma); // row scales = 1, delta = 1

mfem::BlockVector unknown(contact.GetBlockOffsets());
unknown = 0.0; // Set prescribed primal values before evaluating/solving.
```

To use discontinuous degree one instead, **replace the `multipliers`
declaration** above; the contact constructor is unchanged:

```cpp
auto collection = std::make_unique<mfem::L2_FECollection>(
    1, displacement_space.GetMesh()->Dimension() - 1);
plugin::BoundaryMultiplierSpace multipliers(
    *displacement_space.GetMesh(), contact_attributes, std::move(collection));
// collection is now null; multipliers owns it.
```

Degree one has two local DOFs on a segment, three on a triangle (P1), and four
on a quadrilateral (Q1). No continuity is imposed between boundary elements.
The argument immediately after `obstacle` is now `BoundaryMultiplierSpace&`,
not the former attribute array. `gamma` is still compliance, not stiffness;
the optional arguments remain primal row scale, multiplier row scale, and
`delta`, each defaulting to one.

| API | Contract |
| --- | --- |
| `GetMesh()`, `GetSpace()` | Mutable/const references to the owned submesh and FE space; intended for MFEM borrowers such as output grid functions, not mesh/space mutation. |
| `GetParentBoundaryElement(e)` | Original parent **boundary-element** index for submesh element `e`, not a parent mesh-face index. |
| `GetElementDofs(e, dofs)` | Unsigned multiplier DOFs local to exactly one element; no DOF transformations. |
| `CalcShape(e, parentBoundaryPoint, shape)` | Basis values at an original parent **boundary-element reference point**; resizes `shape` if needed. |

Contact's public `GetBlockOffsets()`, `GetContactSubMesh()`,
`GetMultiplierSpace()`, and `GetMultiplier(unknown, coefficients)` are unchanged.
The mesh/space accessors now refer to the borrowed boundary object;
`GetMultiplier` copies coefficients from the caller's monolithic
`[primal true DOFs, multiplier true DOFs]` vector. For initialization or output,
use a `GridFunction` on `multipliers.GetSpace()` and MFEM projection/evaluation
rather than interpreting arbitrary basis coefficients as point values.
For the positive/Bernstein basis, MFEM's default coefficient projection can
sample a monotone approximation rather than reproduce the polynomial exactly;
use `ProjectCoefficientElementL2` when exact representable-field initialization
is required, as in the quadratic-basis regression below.

The parent mesh must outlive `multipliers`. The boundary object owns its
submesh geometry, collection through `unique_ptr`, and FE space; destruction
releases the space before its collection and mesh. Contact borrows the boundary
object, primal operator, extraction, displacement space, obstacle, and optional
integration rule. Destroy contact and other borrowers before their dependencies.
Both `BoundaryMultiplierSpace` and `SemismoothRigidContactOperator` prohibit
copy and move. Temporary extraction operators and obstacles are rejected.

Reconstruct the boundary object and its borrowers after changes to parent or
submesh geometry, topology, attributes, order, or layout; do not call mesh/space
updates through the accessors. There is no automatic AMR solution transfer or
restart facility here. The caller must transfer solution coefficients if
reconstruction is required, and restore both unknown blocks on a rejected step.
Mutable MFEM caches and contact scratch are not reentrant/thread-safe. A returned
gradient reference is invalidated by another contact or wrapped-primal gradient
evaluation.

## Basis-aware contact equations

Let the body dimension be \(d=2\) or \(3\), and let \(e\) be a reference
contact element of dimension \(d_c=d-1\). Define the **column** of local
multiplier basis values \(\mathbf M_q\), coefficient vector
\(\boldsymbol\Lambda_e\), and vector displacement interpolation \(\mathbf N_q\):

$$
\lambda_q=\mathbf M_q^T\boldsymbol\Lambda_e,
\qquad \mathbf x_q=\mathbf X_q+\mathbf N_q\mathbf U_e,
\qquad z_q=\lambda_q-g(\mathbf x_q)/\gamma,
\qquad p_q=\max(0,z_q).
$$

Here \(\mathbf X_q\) is a reference point and \(\mathbf U_e\) contains local
displacement coefficients. The multiplier is positive in compression;
\(g\) is positive in the admissible region,
\(\mathbf n_q=\nabla g(\mathbf x_q)\),
\(\mathbf H_q=\nabla^2g(\mathbf x_q)\), and \(\chi_q=1\) for
\(z_q\geq0\), otherwise zero. Both trial fields and their tests use the same
physical points. With rule weight \(\widehat w_q\), reference boundary metric
\(J_{\Gamma_0,q}\), and \(w_q=\widehat w_q J_{\Gamma_0,q}\), the contact
functional is the repository adaptation

$$
\Pi_c=\frac{\gamma}{2}\sum_{e,q}w_q(p_q^2-\lambda_q^2)
      -\frac12\boldsymbol\Lambda^T S\boldsymbol\Lambda.
$$

Its saddle sign follows stabilized Formulation 1 in Burman, Hansbo, and
Larson's [2016 v1 PDF](https://arxiv.org/pdf/1609.03326v1), p. 4, (2.10),
with source signs \(u_B=-g\), \(\lambda_B=-\lambda_h\); the vector
signed-distance chain rule is a repository derivation, not a source theorem.
Writing \(\mathbf B_q=\mathbf N_q^T\mathbf n_q\), differentiation gives
the unscaled contact coefficient residuals (global insertion understood):

$$
\mathbf r_u=-\sum_{e,q}w_qp_q\mathbf B_q,
\qquad
\mathbf r_\lambda=\gamma\sum_{e,q}w_q(p_q-\lambda_q)\mathbf M_q
                  -S\boldsymbol\Lambda.
$$

On a fixed smooth branch, all four blocks are

$$
K_{uu}=\sum_{e,q}w_q\mathbf N_q^T
       \left(\frac{\chi_q}{\gamma}\mathbf n_q\mathbf n_q^T-p_q\mathbf H_q\right)\mathbf N_q,
$$

$$
K_{u\lambda}=-\sum_{e,q}w_q\chi_q\mathbf B_q\mathbf M_q^T,
\qquad
K_{\lambda u}=-\sum_{e,q}w_q\chi_q\mathbf M_q\mathbf B_q^T,
$$

$$
K_{\lambda\lambda}=\gamma\sum_{e,q}w_q(\chi_q-1)\mathbf M_q\mathbf M_q^T-S.
$$

These are derivatives of the stated quadrature residual, with the active
generalized derivative selected at equality. The reference weights have no
displacement derivative; the curved-obstacle term is \(-p_q\mathbf H_q\).
P0 is exactly the specialization \(\mathbf M_q=[1]\), not a separate kernel.
The multiplier equations enforce basis moments, not pointwise complementarity.

Positions, displacement, and gap have length units; \(\lambda_h,p\) have
stress units (nominal traction per reference area in 3D). Basis functions and
normals are dimensionless, \(\mathbf H\) has inverse-length units,
\(\gamma>0\) has length/stress units and a finite reciprocal, and
\(\delta\geq0\) is dimensionless. Unscaled residual coefficients have units
stress times length\(^{d-1}\) for displacement and length\(^d\) for
multipliers; 2D forces are per out-of-plane thickness. Use consistent units.

Contact applies positive row scales \(s_u,s_\lambda\) once, including the
jump row, and composes the displacement blocks with the borrowed extraction
\(\mathbf u=E\mathbf y\). Before scaling the coupling blocks are transposes;
after scaling they are generally not. The transpose action of the complete
scaled Jacobian must transpose the row scaling too. Equal scales and a
symmetric primal tangent give symmetry, not positive definiteness.

The top-left block is composed by the [lazy operator-sum utility](operator-sums.md).
Each `GetGradient` obtains the current primal Jacobian and constructs the complete
scaled sum; repeated linear-operator applications do not request new gradients.

## Jump matrix and reference mapping

For neighbors \(e,f\) sharing an interior contact-submesh interface \(F\),
set

$$
h_F=\tfrac12\bigl(|e|^{1/d_c}+|f|^{1/d_c}\bigr),\qquad
c_F=\delta\gamma h_F,\qquad
J_F=[\mathbf M_e^T,-\mathbf M_f^T],\qquad
S_F=c_F\int_F J_F^TJ_F\,ds_0.
$$

Here \(|e|\) is the computed reference element measure, \(h_F\) has length
units, and \(S\) is the global assembly of the interface blocks \(S_F\).
This adopts the jump structure of the [v1 PDF](https://arxiv.org/pdf/1609.03326v1),
p. 4, (2.7); the local area-root scale is the repository's choice.
`BuildMultiplierJumpStabilization` integrates the basis products and caches
each unscaled `MultiplierJump.Matrix` with its concatenated `Dofs`. Each
interface is counted once, including active/inactive neighbors. Its two basis
traces use MFEM's submesh `Loc1`/`Loc2` points at the same interface position.
No coefficient-wise difference or separately measured interface coefficient
can replace this integral for a general basis.

For a 2D body, \(F\) is a vertex with measure one; for a 3D body it is an
edge with quadrature-integrated reference metric, not chord length. P0 reduces
to \(c_F|F|\begin{bmatrix}1&-1\\-1&1\end{bmatrix}\), retaining the
original scaling. `AssembleMultiplierJumpStabilization` adds
\(-s_\lambda S\boldsymbol\Lambda\) and \(-s_\lambda S\) to residual and
Jacobian. Nonnegative weights make \(S\) positive semidefinite; matching
traces, including constant functions, are unpenalized. For an arbitrary basis,
this is a statement about functions, not a graph-Laplacian or coefficient
minimum principle.

`BoundaryMultiplierSpace::VerifyElementMaps` runs at construction and checks
parent boundary geometry, identical local vertex ordering through the parent
vertex map, and unique unsigned element-local DOFs. That verified order makes
the original parent **boundary-element-to-submesh** reference map the identity.
In contrast, `VisitElementQuadraturePoints` first maps a parent **mesh-face** rule point
through the inverse boundary/face orientation. It then evaluates displacement,
geometry, and `BoundaryMultiplierSpace::CalcShape` at that original boundary
point. Passing the mesh-face point directly to `CalcShape`, or applying a
second face-orientation map there, would be wrong for nonconstant bases.

`AssembleContact` owns the element loop explicitly: `ResetElementAssembly`,
`VisitElementQuadraturePoints`, then `ScatterContactElement` once per element.
The current displacement and multiplier DOF maps remain valid until scatter;
there is no saved previous-element map, transition sentinel, or final flush.
`ContactPointData` supplies borrowed displacement/multiplier shapes and signed
distance evaluation, together with scalar multiplier, pressure, weight, and
activity. The point helpers read this view and write the local assembly buffers.
Its references are valid only for the synchronous callback, not after the next
point overwrites scratch. Diagnostics use the same per-element visitor and point
view. `mDisplacementShape` distinguishes displacement from multiplier scratch.

## Quadrature, geometry, and constraints

- Parent meshes must be serial, conforming, full-dimensional 2D/3D meshes on
  MFEM's default CPU backend. Attributes must be nonempty, positive, unique,
  present, and select exterior boundary elements; MFEM's boundary-submesh
  topology requirements also apply. Contact requires the **exact same parent
  mesh object** as the displacement space, not merely a geometrically equal mesh.
- NURBS parents are explicitly rejected. If the parent has geometry nodes,
  it must not itself be a `SubMesh`, and its nodes must use uniform-order
  Gauss--Lobatto H1, replicated by spatial dimension. MFEM's submesh transfer
  copies geometry coefficients, not a change of basis; other nodal bases could
  silently change the metric. Nested transfers do not compose immediate-parent
  boundary IDs into root IDs, and discontinuous geometry can duplicate parent
  DOFs where MFEM requires a one-to-one transfer map. These unsupported paths
  fail before submesh creation. These restrictions concern **geometry**, not the
  discontinuous multiplier basis or its choice of Gauss--Lobatto points.
- `BuildIntegrationRules` caches one rule pointer per contact element at
  construction and rebuilds that cache on `SetIntegrationRule`. Defaults use
  adjacent-volume `OrderW()` plus
  `2 * max(displacement order, multiplier order)`, with the existing extra
  order for a `Pk` volume element. Residual, Jacobian, and diagnostics share
  the cached rules through `VisitElementQuadraturePoints` and apply
  `ip.weight * boundaryTransformation->Weight()` once. The default P0 rule is
  retained; reconstruct borrowers after mesh/geometry/space changes as above.
- `SetIntegrationRule` borrows a parent mesh-face-coordinate rule; `nullptr`
  restores the default. An empty rule gives no point terms/samples but still
  leaves jumps. It does **not** change cached jump quadrature. Jump rule order
  is `2 * geometryOrder + 2 + 2 * max(neighbor multiplier orders)`, where
  `geometryOrder` is zero for a vertex, otherwise
  `max(face.OrderW(), min(Elem1.OrderW(), Elem2.OrderW()))`.
  Curved metrics, signed distance, and an active-set split need quadrature
  sensitivity checks; degree counting alone does not ensure resolution.
- Displacement is an H1 vector field with equal local/true sizes and no
  boundary DOF transformations. Essential displacement DOFs must match the
  primal constraints after extraction. Prescribed values enter the gap, but
  essential contact rows and columns are masked. Each face must retain a free
  displacement trace DOF. On a fully assembly-sampled active face, gradient
  assembly tests the masked `mElementConstantNormalCoupling` directly, without
  assuming nodal coefficients or summing multiplier columns. Its norm must
  exceed `sqrt(epsilon)` times its nonzero unmasked norm, where `epsilon` is
  machine epsilon for `mfem::real_t`. This restrictive local constant-mode
  guard is neither a full-rank/inf-sup test nor a necessary
  condition for every globally stabilized system to be nonsingular. Jumps can
  control additional modes; the extraction can also remove variations.

`MinimumMultiplier` and `MaximumMultiplier` sample \(\lambda_q\), not the
coefficient extrema or exact function extrema. No clipping or pointwise
nonnegativity constraint is imposed on multiplier iterates. Positive sampled
values do not guarantee positivity between samples; P0-specific positivity
arguments must not be extended to arbitrary bases. The unscaled diagnostic
`MaximumComplementarityResidual` is \(\gamma\max_q|p_q-\lambda_q|\), not
the norm of the stabilized multiplier equation. Finite gap samples likewise
do not certify continuous nonpenetration.

## Regression map and verification limits

The following constructions are in
[`tests/contact_test.cpp`](../tests/contact_test.cpp). They are assembly/API
checks, not solved higher-order contact benchmarks or general stability proofs.
Executed verification is recorded separately below.

| Regression (suite prefix shown) | What it checks |
| --- | --- |
| `BoundaryMultiplierSpace.DefaultP0AndConfiguredP1Q1MatchParentBoundaryPoints` | Collection ownership, P0/P1/Q1 counts, DOF maps, affine multiplier evaluation and physical-coordinate reconstruction on 2D/3D simplex/tensor-product parents. |
| `BoundaryMultiplierSpace.RejectsUnsupportedCollectionsAndInvalidAttributes` | H1, integral mapping, wrong collection dimension, and empty/duplicate/missing attributes. File-scope `static_assert`s also check lifetime-related constructor and copy/move restrictions. |
| `SemismoothContact.BoundaryP0HasOneMultiplierPerSelectedFace` | Default count/block offsets and mesh/space accessor identity. Existing P0 residual, gamma/delta, equality, and jump regressions remain applicable. |
| `SemismoothContact.BoundaryP1Q1HighOrderTraceMatchesVolumeOracle` | Curved quadratic geometry, order-4 H1 triangles/quads/tets/hexes, both layouts, active/inactive points, default/nonsymmetric rules, physical inversion as an independent multiplier map, directional Jacobian and trace-zero support; jumps disabled. |
| `SemismoothContact.BoundaryP1PlaneAndSphereMixedBlocksTransposeAndAssemblyReset` | Each of the four blocks against centered differences, numerical transpose with unequal scales, both layouts, active/inactive/mixed sampling, and release/reactivation buffer reset. |
| `SemismoothContact.BoundaryP1EssentialMaskMatchesIndependentRowAndColumnElimination` | Essential masking of all blocks and transpose actions, preserving prescribed gap data. |
| `SemismoothContact.BoundaryP1AffineMultiplierMomentsArePreservedUnderRefinement` | Manufactured affine force/multiplier moments on 1, 2, and 4 flat edges; not a solved PDE convergence rate. |
| `SemismoothContact.BoundaryP1Q1JumpUsesTracesAndCurvedInterfaceProducts` | Vertex and curved-edge jump products, sign, every multiplier column, higher-order interface oracle, and zero jump for matching constant/affine traces despite unequal neighboring coefficients. |
| `BoundaryMultiplierSpace.RejectsUnsupportedCurvedParentGeometry` | Rejects discontinuous geometry and curved nested-submesh geometry before MFEM's unsupported transfer path. |
| `SemismoothContact.HigherOrderMultiplierQuadratureAndPositiveBasis` | Degree-two Gauss--Legendre/positive multipliers with degree-one displacement, exact quartic mass moments with default quadrature, active moments, Jacobian, and transpose. |
| `SemismoothContact.HigherOrderPairCanRetainMultiplierNullMode` | Explicit fully active degree-one displacement/degree-four multiplier null mode despite positive jump stabilization; constant multiplier variations still couple. |
| `SemismoothContact.EmptyRuleRetainsJumpsAndRestoresPointAssemblyAndDiagnostics` | Two faces and both layouts: empty quadrature retains only the analytical jump residual/Jacobian, resets diagnostic samples, and restoring the default rule recovers active/inactive point assembly. |

The circle example intentionally still constructs **default P0**. Its
one-multiplier-per-face count and inactive-P0 diagonal preconditioner are
unchanged and must not be reused as a generic higher-order solver design.
No large-scale solve, arbitrary-order/basis validation, uniform inf-sup,
locking, or error-rate evidence is supplied here. Continuous/IGA
multipliers, MPI, nonconforming contact, device kernels, two-body search, and
friction remain outside this implementation.

### A configured pair can still be singular

The null-mode regression is a repository-derived counterexample, not a result
quoted from the source paper. On one flat edge with coordinate `s` in `[0,1]`,
take a degree-four multiplier perturbation and set it to zero on its neighbor:

```text
b(s) = s*(1-s)*(1 - 5*s*(1-s))
b(0) = b(1) = 0
integral_0^1 b(s) ds = integral_0^1 s*b(s) ds = 0
```

Its jump is zero and it is orthogonal to every linear normal displacement
test. On the fully active branch the multiplier mass block is zero, so
`J * [0, b] = 0` even for positive `delta`. The constant-mode guard cannot
detect this mode. Configurability does not imply an admissible pairing; use
appropriate displacement/multiplier spaces and assess their stability rather
than increasing `delta` as a universal remedy.

### Executed verification (2026-09-09)

The working tree was based on `bf40b5f6e13152cd47ddf550066e3dd5e909f0f7`,
including pre-existing uncommitted contact changes and this extension. The
existing `build/debug` and `build/release` configurations used GCC 15.2.0,
MFEM 4.9.1 in double precision, and the sibling MFEM installations selected by
the project presets. These were incremental builds, not clean configurations.

```bash
cmake --build build/debug --parallel 4
ctest --test-dir build/debug --output-on-failure
cmake --build build/release --target contact_test penalty_contact --parallel 4
ctest --test-dir build/release \
  -R '^(RigidObstacle|BoundaryMultiplierSpace|PenaltyContact|SemismoothContact)\.' \
  --output-on-failure
```

All 159 enabled Debug tests passed, including the existing one-/two-rank tests
of other models. All 44 enabled Release contact/example checks passed. Contact
itself remains serial. The two opt-in assembly microbenchmarks were disabled in
CTest; the semismooth one was run explicitly as described below. Single-precision
MFEM and new MPI/device contact paths were not tested. The SLEPc-dependent
`eigenbuckling` target was skipped by configuration because SLEPc is disabled.

The existing P0 trace microbenchmark used 3-by-3-by-3 hexahedra, H1 displacement
order four, `byVDIM`, all boundary attributes, a radius-six sphere centered at
`(-2,-2,-2)`, zero displacement, constant multiplier 100, `gamma=1/17`, default
unit row scales and `delta=1`, and default quadrature. It measures 20 residual/
Jacobian assembly pairs per sample after warm-up, not a nonlinear solve.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build/release/bin/contact_test --gtest_also_run_disabled_tests \
  --gtest_filter=SemismoothContact.DISABLED_TraceAssemblyBenchmark
```

Before the extension the five samples were 7.07583, 6.97211, 6.96401, 6.95615,
and 6.98063 ms/pair (median 6.97211). Isolated reruns after extension used the
same command with `--gtest_repeat=3`; their per-run medians were 7.32547,
7.32266, and 7.67335 ms/pair (pooled median 7.37123). The printed residual norm
remained 29.7788. This suggests roughly 6% P0 overhead in this small benchmark,
not a speedup. Timing varies, CPU affinity was not controlled, and peak memory
and scaling were not measured; do not interpret these samples as a universal
regression threshold.

### Element-loop readability follow-up (2026-09-09)

A subsequent refactor replaced the whole-boundary point callback with the
explicit per-element sequence described above and a borrowed `ContactPointData`
view. It removed the previous-element sentinel and duplicated DOF maps, without
changing the public API, equations, quadrature, or accumulation order of point
contributions. The new empty-rule regression checks that scattering zero local
buffers still retains only jumps and that restoring quadrature restores the
residual, Jacobian, and diagnostics.

Using the same build/test commands and configuration above, all 160 enabled
Debug tests and all 45 enabled Release contact/example checks passed after this
follow-up. The formulation PDF also rebuilt successfully. The opt-in P0 trace
benchmark command above gave five before-refactor samples of 7.72433, 7.66078,
7.65246, 7.65964, and 7.66849 ms/pair (median 7.66078). After-refactor samples
were 7.79862, 7.82534, 7.87205, 7.80719, and 7.54763 (median 7.80719), with the
same printed residual norm 29.7788. These short measurements do not establish
a speedup; the timing, precision, and scalability limits above still apply.

## Source correspondence

Erik Burman, Peter Hansbo, and Mats G. Larson, *Augmented Lagrangian finite
element methods for contact problems*, arXiv:1609.03326v1, 12 September 2016,
26-page [PDF](https://arxiv.org/pdf/1609.03326v1)
([record](https://arxiv.org/abs/1609.03326v1)). **Reading scope:** source
identity and PDF pp. 1--4 were already verified through Zotero `ZAV78EWP` in
the preceding discussion; this note uses those supplied locators, not a new
source retrieval. Section 2, p. 3 defines continuous H1 degree \(k\) and
discontinuous L2 degree \(l=k-1\); p. 4, (2.7) gives the multiplier jump form,
and (2.10) gives the stabilized saddle sign. The API adopts the multiplier
basis/jump structure but does not enforce that degree pairing.

The source analyzes a scalar elliptic model on fitted quasiuniform simplicial
meshes, with exact integration across the active interface and lifting/parameter
assumptions. It is not a vector curved-contact theorem. The reference mapping,
local \(h_F\), vector linearization, finite quadrature, and constraints above
are repository choices/derivations. Prior HTML equations (9) and (12) have
different numbering. The journal citation, *ESAIM: M2AN* **53**(1), 173--195
(2019), [DOI 10.1051/m2an/2018047](https://doi.org/10.1051/m2an/2018047),
is metadata only here, not the source of the PDF equation numbers or a proof
of stability for this extension.

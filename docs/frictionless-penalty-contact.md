# Frictionless penalty contact with a rigid obstacle

## Status and design goal

This feature implements unilateral, frictionless contact between a deformable
body and a smooth analytic rigid obstacle. The first implementation uses a
penalty regularization and has no history variables. It is intended both as a
small contact-mechanics reference problem and as a building block for later
coupled examples.

The obstacle is analytic rather than a second finite-element mesh. Contact
search, two-body contact, friction, adhesion, cohesive bonding, and an
augmented-Lagrangian multiplier update are outside the implemented scope.

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

## Discrete evaluation and ownership

At each boundary-face quadrature point, the integrator:

1. sets the face and neighboring-element integration points;
2. evaluates the volume finite-element shape functions on the face;
3. maps the face point through the mesh transformation to obtain \(\mathbf X\);
4. interpolates displacement and forms \(\mathbf x=\mathbf X+\mathbf u_h\);
5. evaluates and validates \(g\), \(\mathbf n\), and \(\mathbf H\);
6. applies the residual or tangent formula with
   `ip.weight * transformation.Weight()` exactly once.

The default integration order follows MFEM's boundary-face heuristic,
`Elem1->OrderW() + 2 * element.GetOrder()` with one additional order for a
`Pk` element. A finite rule is not exact for a general signed distance or for a
face crossed by the active-set boundary. Use `SetIntegrationRule()` and check
quadrature sensitivity for such problems; the same rule must be used for the
residual and tangent.

The implementation supports full-dimensional H1 displacement fields in 2D and
3D under MFEM legacy assembly. Element-local vector values use MFEM's
component-major layout. Embedded manifolds, ND/RT fields, partial assembly,
matrix-free assembly, and device kernels are unsupported. Applications must
leave `mfem::NonlinearForm` at its default `AssemblyLevel::LEGACY`: current MFEM
matrix-free form extensions do not visit boundary-face integrators and can omit
their contribution rather than dispatching a callback that this integrator
could reject.

The integrator stores a non-owning reference to the obstacle. The obstacle must
outlive the nonlinear form because MFEM owns integrators passed to
`AddBdrFaceIntegrator`, while the integrator continues to borrow the obstacle.
A custom integration rule and boundary marker are likewise owned by the caller
and must remain alive as required by MFEM. The integrator reuses mutable
assembly scratch and is not reentrant; threaded assembly requires one instance
per thread.

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
residual.AddBdrFaceIntegrator(
    new plugin::FrictionlessPenaltyContactIntegrator(obstacle, penalty),
    contact_boundary_marker);
```

MFEM does not provide a boundary-face energy callback for
`NonlinearFormIntegrator`. Consequently, adding this integrator makes
`NonlinearForm::GetEnergy()` unavailable. Residual-based Newton methods remain
usable; energy-based globalization would require a deliberate form-level
boundary-face energy extension.

## Penalty selection

A useful mesh-dependent starting value is

$$
\kappa = \gamma\frac{E_{\mathrm{eff}}}{h_c},
$$

where \(h_c\) is a representative contact-face size and \(\gamma\) is a
dimensionless factor. This is a numerical enforcement parameter, not a
material property. Increasing \(\gamma\) reduces penetration but worsens the
conditioning of the discrete system. The benchmark therefore reports
penetration, contact pressure, contact resultant, and nonlinear convergence,
and is intended to be repeated for several values such as
\(\gamma=1,10,100\).

## Verification

Focused regression tests cover:

- plane and circle/sphere signed-distance values and derivatives;
- inactive contact and the active-at-zero tangent convention;
- residual sign, reference-face scaling, and zero tangential response for a
  penetrated plane;
- a curved-obstacle Jacobian against a centered directional finite difference,
  including tangential perturbations that detect a missing \(g\mathbf H\) term;
- equivalent assembled resultants for `mfem::Ordering::byVDIM` and
  `mfem::Ordering::byNODES`.

The standalone 2D benchmark drives the complete bottom boundary of a
rectangular elastic block upward while constraining horizontal motion on the
left boundary. The top boundary contacts an off-center rigid circle. The
default circle has center \((0.65,1.26)\), radius \(0.25\), and initial vertical
clearance \(0.01\) above a unit-square block; the bottom displacement ramps to
\(0.02\). The benchmark checks initial separation, prescribed-displacement
enforcement, nonlinear convergence, active contact, a downward contact
resultant, and localization within the circle's horizontal projection. The
circle geometry and displacement are configurable from the command line.

The `-r/--refine-level` option applies uniform mesh refinement before the
finite-element spaces are created. The benchmark updates \(h_c\), and therefore
\(\kappa=\gamma E/h_c\), after every refinement. Its ParaView output registers
the planar displacement as \((u_x,u_y,0)\) so vector filters work consistently.
It also writes a closed polyline approximation of the analytic rigid circle as
the separate `penalty_contact_obstacle` dataset. Load both `.pvd` files in
ParaView; `Warp By Vector` then moves only the deformable `penalty_contact`
dataset and leaves the obstacle fixed.

## Planned augmented-Lagrangian extension

The augmented-Lagrangian version is deliberately deferred to a separate
change. Its planned staggered update is

$$
p_n^{(k)} = \max\left(0,\lambda^{(k)}-\kappa g(\mathbf u^{(k+1)})\right),
\qquad
\lambda^{(k+1)} = p_n^{(k)}.
$$

Each outer multiplier iteration will hold \(\lambda^{(k)}\) fixed during an
inner displacement Newton solve. Multipliers will be quadrature-point state and
will commit only after the complete load step converges; rejected steps must
restore displacement, load factor, material history, and contact multipliers.
No part of that stateful algorithm is included in the penalty implementation.

## References and provenance

1. Wriggers, P. (2006), *Computational Contact Mechanics*, 2nd edition,
   Springer, Berlin, Heidelberg.
   [doi:10.1007/978-3-540-32609-0](https://doi.org/10.1007/978-3-540-32609-0).
   Section 2.1.3 supplies the penalty regularization used here; the notation is
   adapted to this repository's positive-gap convention and a rigid analytic
   obstacle.

The reference-boundary residual and the complete curved-obstacle tangent above
are derived in this document by first and second variation of the stated
penalty potential. The active-at-zero generalized tangent and the exact API
validation rules are implementation choices of this repository rather than
claims attributed to the reference.

# Thermal expansion kinematics

## Design goal

Temperature changes alter the local stress-free configuration; they are not a
property of an elastic constitutive law. The material therefore receives only
the mechanical part of the deformation, while a separate kinematic field owns
the thermal expansion model.

This separation allows the same elastic material to be used with no thermal
load, an isotropic thermal load, or another imposed stress-free deformation.
It also prevents a small-strain additive formula from being silently used in a
finite-deformation analysis.

## Kinematic model

The implementation has two explicitly different paths:

| Integrator mode | Decomposition | Constitutive input |
| --- | --- | --- |
| Small deformation | $\boldsymbol\varepsilon_e = \boldsymbol\varepsilon - \boldsymbol\varepsilon_0$ | Mechanical strain $\boldsymbol\varepsilon_e$ |
| Finite deformation | $\mathbf F = \mathbf F_e\mathbf F_0$ | Elastic gradient $\mathbf F_e = \mathbf F\mathbf F_0^{-1}$ |

Here $\mathbf F_0$ maps the initial reference configuration to a local
stress-free configuration. For thermal expansion it is denoted by
$\mathbf F_\theta$. This intermediate configuration can be locally
incompatible when temperature varies in space; the decomposition remains a
pointwise constitutive construction rather than a claim that a global
stress-free mesh exists.

### Small deformation

For isotropic expansion with a constant coefficient of thermal expansion
$\alpha$,

$$
\boldsymbol\varepsilon_\theta
  = \alpha (T-T_{\mathrm{ref}})\mathbf I,
\qquad
\boldsymbol\varepsilon_e
  = \boldsymbol\varepsilon-\boldsymbol\varepsilon_\theta.
$$

`NonlinearElasticityIntegrator::setNonlinear(false)` selects this path. The
material must explicitly support a mechanical-strain input. This is currently
true for `IsotropicElasticMaterial` and false for deformation-gradient-only
materials such as `NeoHookeanMaterial`.

### Finite deformation

The finite-deformation path uses

$$
\mathbf F = \mathbf F_e\mathbf F_\theta,
\qquad
\mathbf F_e = \mathbf F\mathbf F_\theta^{-1}.
$$

For isotropic thermal expansion,

$$
\mathbf F_\theta = \vartheta(T)\mathbf I,
\qquad
\vartheta(T)
  = \exp\left(\int_{T_{\mathrm{ref}}}^{T}\alpha(\tau)\,d\tau\right).
$$

`IsotropicThermalExpansion` currently assumes $\alpha$ is constant over the
temperature interval, giving

$$
\vartheta(T) = \exp\left(\alpha(T-T_{\mathrm{ref}})\right).
$$

The load factor ramps the temperature change inside these expressions. The
small-strain and finite-strain formulas consequently agree to first order.

## Finite-element transformation

The constitutive law is evaluated relative to the local stress-free
configuration. Shape gradients and the integration measure must be transformed
with it; changing only the deformation gradient would not produce a consistent
residual or tangent.

For a shape function $N_a$,

$$
\nabla_\theta N_a = \mathbf F_\theta^{-T}\nabla_0 N_a,
\qquad
dV_\theta = J_\theta\,dV_0,
\qquad
J_\theta = \det\mathbf F_\theta.
$$

`IntegrationPointStorage` stores gradients with row-vector layout, so the code
implements the equivalent matrix operation

$$
\mathbf G_\theta = \mathbf G_0\mathbf F_\theta^{-1}.
$$

The internal force and material tangent are then assembled as

$$
\mathbf r_e
  = \int_{\Omega_0}
      J_\theta\mathbf B_e^T\mathbf S_e\,dV_0,
$$

$$
\mathbf K_{\mathrm{mat}}
  = \int_{\Omega_0}
      J_\theta\mathbf B_e^T\mathbb C_e\mathbf B_e\,dV_0,
$$

with the geometric tangent using $\mathbf G_\theta$ and $\mathbf S_e$. This is
equivalent to mapping the elastic second Piola stress back to the initial
reference configuration:

$$
\mathbf S
  = J_\theta\mathbf F_\theta^{-1}
      \mathbf S_e\mathbf F_\theta^{-T}.
$$

Because `StressCoefficient` evaluates Cauchy stress from $\mathbf F_e$ and
$\mathbf S_e$, no additional stress transformation is needed for output.

## API and ownership

```cpp
mfem::ConstantCoefficient reference_temperature(20.0);
mfem::ConstantCoefficient target_temperature(120.0);

plugin::IsotropicThermalExpansion thermal_expansion(
    coefficient_of_thermal_expansion,
    target_temperature,
    reference_temperature);

IsotropicElasticMaterial material(elastic_modulus, poisson_ratio);
plugin::NonlinearElasticityIntegrator integrator(material, point_storage);
integrator.AddStressFreeDeformation(thermal_expansion);
```

`StressFreeDeformationModel` stores non-owning references. Every registered
field and every referenced MFEM coefficient must outlive the integrator or
stress coefficient that uses it.

Multiple small-strain contributions are added. Multiple finite-deformation
contributions are multiplied in insertion order, because multiplication is not
commutative for general anisotropic deformations.

## Scope and limitations

- The current implementation is one-way thermomechanical coupling. A monolithic
  temperature solve would also require residual and tangent blocks with respect
  to temperature.
- `IsotropicThermalExpansion` treats the supplied $\alpha$ as constant over the
  interval from $T_{\mathrm{ref}}$ to $T$. A temperature-dependent coefficient
  needs a model that evaluates the integral of $\alpha(T)$.
- A stress-free deformation must be finite, invertible, and
  orientation-preserving.
- The two-dimensional formulation retains a three-dimensional constitutive
  state and therefore follows the existing plane-strain convention. It rejects
  stress-free gradients that couple in-plane and out-of-plane directions.
- Temperature dependence of elastic moduli remains a constitutive concern and
  can be represented by spatial or temperature-backed MFEM coefficients. The
  thermal expansion kinematics themselves remain outside the material law.

## Verification requirements

The regression tests cover these invariants:

- constrained heating creates a nonzero residual;
- uniform free expansion creates zero stress in both kinematic modes;
- the small-strain thermal load scales with the continuation load factor;
- the small-strain material tangent is unchanged by an imposed eigenstrain;
- the finite-strain analytical tangent agrees with a finite-difference
  directional derivative for a Neo-Hookean material.

## References

1. Lu, S. C. H., and Pister, K. S. (1975), “Decomposition of deformation and
   representation of the free energy function for isotropic thermoelastic
   solids,” *International Journal of Solids and Structures*, 11(7–8), 927–934.
   [doi:10.1016/0020-7683(75)90015-3](https://doi.org/10.1016/0020-7683(75)90015-3).
2. Vujošević, L., and Lubarda, V. A. (2002), “Finite-strain thermoelasticity
   based on multiplicative decomposition of deformation gradient,”
   *Theoretical and Applied Mechanics*, 28–29, 379–399.
   [Open-access article](https://doi.org/10.2298/TAM0229379V). Equations
   37, 41, 45, 50, and 59 directly support the decomposition, elastic strain,
   volume mapping, thermal stretch, and stress mapping used here.
3. Darijani, H., and Naghdabadi, R. (2013), “Kinematics and kinetics modeling
   of thermoelastic continua based on the multiplicative decomposition of the
   deformation gradient,” *International Journal of Engineering Science*, 62,
   56–69.
   [doi:10.1016/j.ijengsci.2012.07.001](https://doi.org/10.1016/j.ijengsci.2012.07.001).
4. Hartmann, S. (2012), “Comparison of the multiplicative decompositions
   $\mathbf F=\mathbf F_\theta\mathbf F_M$ and
   $\mathbf F=\mathbf F_M\mathbf F_\theta$ in finite-strain thermoelasticity,”
   Technical Report FAK3-12-01, Clausthal University of Technology.
   [Institutional PDF](https://dokumente.ub.tu-clausthal.de/servlets/MCRFileNodeServlet/import_derivate_00000071/FAK3-12-01.pdf).

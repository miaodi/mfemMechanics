# Copper thermal protrusion example

## Purpose and scope

`pCuProtrusion` is an MPI example for the heating stage of a dished copper
feature. It combines the repository's infinitesimal, rate-independent J2 model
with prescribed isotropic thermal expansion. It is a mechanics calculation,
not a heat-transfer solve: a spatially uniform temperature change is ramped by
dimensionless pseudo-time from zero to one.

The two-dimensional analysis is plane strain. It retains the three-dimensional
constitutive state, including out-of-plane stress and plastic strain. The model
does not include a dielectric, a second copper feature, contact, bonding,
temperature-dependent properties, creep, viscoplasticity, grain structure, or
finite-strain kinematics. Those omissions make this an initial protrusion
experiment rather than a predictive Cu-Cu hybrid-bonding process model.

## Geometry and boundary conditions

The default Gmsh mesh is `data/cu_dished_2d.msh`. The example requires exactly
these physical attributes:

```text
cu_dished_2d.msh: x-y section (not to scale)

                              y
                              ^
       (x=-5, y=5)            |             (x=5, y=5)
                    +\___________________/+
                    |  dished_top (13)    |   center: y=4.5
          side (12) |                     | side (12)
          fixed     |     copper (1)      | fixed
                    |                     |
                    +---------------------+----> x
                         bottom (11), fixed
                    x=-5       y=0       x=5
```

`data/cu_dished_3d.msh` is the corresponding cylindrical solid. Its radial
section has the same dimensions and dish profile:

```text
cu_dished_3d.msh: radial section and top view (not to scale)

 radial section                         top view

                 z                             y
                 ^                             ^
       r=-5      |       r=5              .----|----.
          +\___________/+                /     |     \
          | top (13)    |               |      +-----> x
 side (12)| copper (1)  |side (12)       \           /
          +-------------+----> r           '-------'
            bottom (11)                   outer radius = 5
                z=0                      dished top surface (13)

 top rim: z=5; top center: z=4.5; bottom, side, and top use attributes
 11, 12, and 13, respectively.
```

The current `pCuProtrusion` executable is a 2D plane-strain example and rejects
`cu_dished_3d.msh`; the 3D mesh is included as a geometry reference for a future
three-dimensional driver.

| Dimension | Attribute | Physical name | Treatment |
| --- | ---: | --- | --- |
| Domain | 1 | `copper` | J2 copper material |
| Boundary | 11 | `bottom` | `u_x = u_y = 0` |
| Boundary | 12 | `side` | `u_x = u_y = 0` on both sides |
| Boundary | 13 | `dished_top` | Traction-free |

Both side curves share attribute 12. The top endpoints also lie on the fixed
side boundary, so their displacement is zero. The reported protrusion response
is the maximum vertical displacement over all true degrees of freedom on
boundary 13; it is an uplift metric, not the final dish depth.

Mesh coordinates and reported displacements use the mesh's length unit. Stress
inputs use MPa, temperatures use one consistent Celsius or Kelvin scale, and
CTE uses `1/K`. Only temperature differences enter the thermal strain. The
two-dimensional residual is per unit out-of-plane thickness in the chosen
length unit.

## Model and loading

At pseudo-time `s` in `[0,1]`, the prescribed temperature and thermal strain
are

$$
T(s)=T_{ref}+s(T_{target}-T_{ref}),
\qquad
\boldsymbol\varepsilon_\theta(s)
  =s\alpha(T_{target}-T_{ref})\mathbf I.
$$

The J2 material receives the mechanical strain

$$
\boldsymbol\varepsilon_m
  =\operatorname{sym}(\nabla\mathbf u)
   -\boldsymbol\varepsilon_\theta.
$$

Residual and consistent tangent assembly use the same quadrature points and
the same committed plastic state. Each converged thermal increment commits the
trial J2 history; a failed increment restores the displacement and history,
reduces the pseudo-time step, and retries. See
[Infinitesimal J2 plasticity](j2-plasticity.md) and
[Thermal expansion kinematics](thermal-expansion-kinematics.md) for the
constitutive equations, linearization, references, and verification.

## Material inputs

The command-line defaults are deliberately an illustrative solver dataset:

| Option | Default | Unit |
| --- | ---: | --- |
| `--youngs-modulus` | 110000 | MPa |
| `--poisson-ratio` | 0.34 | 1 |
| `--yield-stress` | 70 | MPa |
| `--hardening-modulus` | 1000 | MPa |
| `--thermal-expansion` | 16.5e-6 | 1/K |
| `--reference-temperature` | 25 | temperature |
| `--target-temperature` | 300 | temperature |

`pCuProtrusion` uses this value as the linear isotropic modulus $H$ and selects
zero kinematic modulus $C$.

These values are not a calibrated material card and must not be used as
evidence for a manufacturing process. Copper yield and hardening depend
strongly on deposition, grain size, annealing, strain rate, and temperature.
The present J2 API accepts spatially varying MFEM coefficients but has no
temperature argument for constitutive properties. Keep process-specific data
outside the constitutive implementation and pass the selected values explicitly.
A material database should be added only with unit metadata, provenance, valid
temperature and rate ranges, and an interpolation policy.

## Build and run

The target requires an MFEM build with MPI and MUMPS:

```bash
cmake --build build/debug --target pCuProtrusion --parallel
mpirun -np 2 build/debug/bin/pCuProtrusion \
  -m data/cu_dished_2d.msh \
  -steps 20 \
  -no-output
```

Use `-output` to write parallel displacement and cell-averaged committed
equivalent plastic strain under `ParaView/pCuProtrusion`. Each accepted step
prints the prescribed temperature, maximum top displacement, and the global
maximum committed equivalent plastic strain at quadrature points. CTest runs
the smoke case with both one and two MPI ranks.
Refine only before loading with `--refine-serial` and `--refine-parallel`;
plastic-history transfer during adaptive mesh refinement is not implemented.

For reproducible studies, record the source revision, MFEM precision and
configuration, rank count, mesh units, all material and temperature options,
finite-element order, refinements, nominal load steps, and the reported uplift
and plastic-strain metrics.

# mfemMechanics

`mfemMechanics` is a research-oriented collection of solid-mechanics materials,
finite element integrators, nonlinear solvers, and examples built on
[MFEM](https://mfem.org/). The repository includes cohesive-zone, phase-field,
thermal, plasticity, buckling, and large-deformation experiments.

## Capabilities

- Hyperelastic and phase-field material models.
- Nonlinear volume, boundary, interface, and composite solid-shell integrators.
- Newton, adaptive load stepping, and arc-length solvers.
- Analytic and automatic-differentiation exponential cohesive-zone laws in 2D
  and 3D.
- Irreversible cohesive history with accepted-step commit, failed-step rollback,
  step-size cutback, and nested-solver handling.
- Infinitesimal associative J2 plasticity with linear isotropic hardening, an
  analytic consistent tangent, and accepted/trial integration-point history.
- Serial and MPI examples with optional OpenMP, MUMPS, PETSc, and SLEPc targets.

## Requirements

- A C++17 compiler.
- CMake 3.20 or newer.
- An MPI C++ implementation.
- An installed MFEM CMake package that exports the `mfem` target.

Eigen 3.4, autodiff 1.1.2, and GoogleTest 1.17.0 are found as packages when
available. Otherwise, CMake fetches pinned versions during configuration.
OpenMP is optional. PETSc and SLEPc examples are enabled only when the installed
MFEM package reports the corresponding capabilities.

## Build

For a portable out-of-tree Debug build, point `CMAKE_PREFIX_PATH` at the MFEM
installation:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=/path/to/mfem/install/debug
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure
```

The committed presets match the sibling dependency layout used for this
workspace: `../mfem/install/{debug,release}`, a sibling cuDSS archive, and a
CUDA toolkit under `$HOME/.local`. With that layout, use:

```bash
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug
```

Replace `debug` with `release` for the optimized configuration. Executables are
written to `build/<configuration>/bin`.

## Target Gates

| Requirement | Targets |
| --- | --- |
| Always available | `test3`, `ex2`, `heat_dynamic`, `heat_static`, `playMesh`, `j2_tensile` |
| MFEM with SuiteSparse | `test2`, `exec`, `beam`, `block`, `postBuckling2D` |
| MFEM with MPI | `ex2p`, `pblock`, `pPhaseField_shear` |
| OpenMP C++ and MFEM with SuiteSparse | `PhaseField_shear`, `czm`, `czm2` |
| MFEM with MPI and MUMPS | `beamParallel`, `pCuProtrusion` |
| MFEM with MPI and PETSc | `petchbuckle`, `postBuckling3D1`, `postBuckling3D2`, `thermalStrain`, `czm2p` |
| MFEM with MPI, PETSc, and SLEPc | `eigenbuckling` |

CMake reports why any capability-gated target is skipped.

## J2 Plasticity

`j2_tensile` prescribes vertical displacement on the top of a plane-strain
rectangular bar while fixing its bottom, reporting displacement and committed
equivalent plastic strain and optionally writing ParaView output:

```bash
(cd build/debug/bin && ./j2_tensile -r 1 -lr 1 -steps 20 -disp 0.01 -no-output)
```

`-r` applies uniform refinement and `-lr` further refines elements touching the
horizontal centerline.

See [Infinitesimal J2 plasticity](docs/j2-plasticity.md) for equations, Voigt
conventions, state lifecycle, limitations, verification, and references.

`pCuProtrusion` uses MUMPS on an MPI-partitioned dished copper mesh. It clamps
the bottom and both sides and applies a prescribed uniform temperature ramp
through isotropic CTE and the J2 material:

```bash
mpirun -np 2 build/debug/bin/pCuProtrusion -steps 20 -no-output
```

The default Cu inputs are illustrative CLI values, not a calibrated material
database. See [Copper thermal protrusion](docs/cu-protrusion.md) for boundary
attributes, units, assumptions, outputs, and model limitations.

## Cohesive History

`ExponentialCZMIntegrator` and `ExponentialADCZMIntegrator` use quadrature-point
committed and trial state. Normal and tangential unloading stiffness cannot
recover after an accepted maximum, including changes in mode mix. Solver
iterations update trial state only; accepted increments commit it, while failed
or exceptional increments roll it back.

Compression is excluded from normal cohesive history and traction. Add a
separate contact formulation when compressive resistance is required. Normal
damping acts only during opening, while tangential damping follows the signed
accepted separation increment. `ExponentialRotADCZMIntegrator` remains
reversible and accepts only zero damping until generalized rotating history is
defined.

See [Integration-point state](docs/integration-point-state.md) for typed state
composition, lifecycle rules, and migration from the previous `AnyMap` API.

## Repository Layout

- `material/`: constitutive material models.
- `fem/`: finite element integrators, cohesive laws, AMR support, postprocessing,
  and nonlinear solvers.
- `examples/`: supported mechanics examples grouped by problem type.
- `tests/`: GoogleTest-based numerical and cohesive-history regression tests.
- `data/`: meshes, geometry sources, and solver option files.
- `cmake/`: dependency helpers.

# AGENTS.md

These instructions apply to the entire repository. There are no more-specific
agent instruction files below this directory.

## Mission and priorities

`mfemMechanics` is a C++17 research library and collection of MFEM examples for
solid-mechanics exploration. Existing code covers linear and nonlinear
elasticity, hyperelasticity, phase-field fracture, cohesive zones, thermal
stress-free deformation, buckling, AMR, and nonlinear/load-continuation
solvers. Plasticity, viscoplasticity, contact, diffusion, and coupled physics
are valid future directions, but an experiment is not a supported capability
until its assumptions, discrete equations, tests, and build target agree.

Prioritize, in order:

1. mathematically and numerically correct behavior;
2. explicit assumptions, references, and reproducible verification;
3. high performance on measured hot paths without obscuring the formulation;
4. small, reviewable changes that preserve useful research flexibility.

## Repository map and sources of truth

- `CMakeLists.txt` and the subdirectory CMake files are authoritative for
  dependencies, targets, and capability gates. Prefer them over stale comments
  or example defaults.
- `material/` owns constitutive response. `ElasticMaterial` is the current base;
  `IsotropicElasticMaterial`, `NeoHookeanMaterial`, and
  `PhaseFieldElasticMaterial` are representative implementations. The CMake
  target is `mfemMechanics::matlib`.
- `fem/` owns MFEM integrators, `IntegrationPointStorage`, cohesive laws,
  stress-free kinematics, postprocessing, AMR, and Newton/arc-length/adaptive
  solvers. The CMake target is `mfemMechanics::femplugin`.
- `util/` owns scalar/Eigen aliases, Voigt/tensor helpers, printing, and small
  utilities (`mfemMechanics::util`). `SymmetricEigensolver3x3.hpp` and
  `CircularBuffer.hpp` are imported code; retain their license notices and do
  not casually restyle or rewrite them.
- `tests/` contains GoogleTest executables registered through CTest. Numerical
  tangent, precision-aware, cohesive-history, and thermal-kinematics tests here
  are the closest testing examples.
- `examples/` contains CMake-supported serial and MPI applications grouped by
  problem. Empty `blockOperator/` and `misc/`, and source files not named by a
  CMake target, are not implicitly supported examples.
- Root `*.cpp` files are standalone/legacy experiment entry points selected by
  the root CMake file. `Plugin.h` is the current umbrella header.
- `data/` contains tracked meshes, Gmsh geometry sources, and PETSc/SLEPc option
  files. Treat them as test/problem inputs, not as disposable generated output.
- `docs/thermal-expansion-kinematics.md` is the model for documenting a
  mathematical feature: conventions, weak-form implications, limitations,
  verification, and precise references.
- `build/`, `.cache/`, and root `compile_commands.json` are generated and
  ignored. Never edit or commit their contents as source.

Several paths remain deliberately incomplete: `NeoHookeanType::Poly2`, the 3D
rotating cohesive law, and some energy-evaluation methods. Keep unsupported
branches explicit and fail fast; do not advertise them as implemented merely
because an enum, class, or example stub exists. In particular, a placeholder
`GetElementEnergy()` that returns zero means “unavailable,” not zero physical
energy; do not use it for energy checks or globalization until it is implemented
and tested.

## Configure, build, test, and run

Requirements are CMake 3.20+, a C++17 compiler, an MPI C++ implementation, and
an installed MFEM CMake package exporting target `mfem`. Eigen 3.4, autodiff
1.1.2, and GoogleTest 1.17.0 are found as packages or fetched at pinned commits
during configuration; a package-free configure therefore needs network access.
OpenMP is optional. Other solver capabilities come from the installed MFEM.

Portable Debug workflow from the repository root:

```bash
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH=/path/to/mfem/install/debug
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure
```

The committed presets encode this workspace's sibling MFEM/cuDSS layout and a
CUDA toolkit under `$HOME/.local`; use them only when that layout exists:

```bash
cmake --preset debug
cmake --build --preset debug --parallel
ctest --preset debug

cmake --preset release
cmake --build --preset release --parallel
ctest --preset release
```

For a focused edit, build the owning test executable and use the discovered
CTest names:

```bash
cmake --build build/debug --target czm_history_test --parallel
ctest --test-dir build/debug -R 'CZMHistory' --output-on-failure
./build/debug/bin/czm_history_test --gtest_filter='CZMHistory.*'
```

Executables are placed in `build/<configuration>/bin`. Many examples contain
historical relative defaults, so pass repository-root input paths explicitly
and disable interactive visualization in unattended runs, for example:

```bash
./build/debug/bin/ex2 -m "$PWD/data/twoElementTensile.mesh" -no-vis
mpirun -np 2 ./build/debug/bin/ex2p -m "$PWD/data/twoElementTensile.mesh" -no-vis
```

Examples can write `ParaView/`, VTK files, or solver output in the current
directory. Run them from a disposable working directory when those artifacts
are not part of the change.

Target availability is configuration-dependent:

- Baseline application targets are `test3`, `ex2`, `heat_dynamic`,
  `heat_static`, and `playMesh`; the four test executables are added when
  `BUILD_TESTING=ON`.
- SuiteSparse gates `test2`, `exec`, `beam`, `block`, and `postBuckling2D`.
- MFEM MPI gates `ex2p`, `pblock`, and `pPhaseField_shear`; OpenMP plus
  SuiteSparse gates `PhaseField_shear`, `czm`, and `czm2`.
- MPI plus MUMPS gates `beamParallel`; MPI plus PETSc gates `petchbuckle`,
  `postBuckling3D1`, `postBuckling3D2`, `thermalStrain`, and `czm2p`; SLEPc is
  additionally required for `eigenbuckling`.

Read configuration skip messages rather than assuming a missing target is a
source failure. There is no repository lint, install, or benchmark target.

### Experiment reproducibility and support status

- Record the source revision and configure command/preset; MFEM precision,
  version, and enabled capabilities; executable; mesh and required attributes;
  order/refinement; parameters and units; load/time increments; solver option
  file, tolerances, and iteration limits; ranks/threads/device; random seed; and
  the convergence and output quantities used for comparison.
- Many examples encode one mesh's material and boundary attribute numbers
  directly. Validate mesh dimension and all required attributes before indexing
  marker/property arrays. A different mesh with the same file format is not
  necessarily a compatible problem definition.
- Promote an exploratory source to a supported capability only after adding it
  to CMake with accurate feature gates, a reproducible noninteractive run,
  focused regression tests, and documentation of its mathematical scope. Until
  then label results and limitations as experimental.

## C++ and MFEM conventions

- Follow the root `.clang-format`: four spaces, no tabs, Allman braces, left
  pointer alignment, and a 120-column limit. Run `clang-format -i` only on
  touched C/C++ files; avoid repository-wide formatting churn.
- Match nearby naming while improving new code locally: classes are generally
  PascalCase, implementation files pair `.h`/`.cpp`, library headers use
  `#pragma once`, and FEM extensions live in namespace `plugin`. Do not perform
  a broad namespace or API rename as part of a physics change.
- Use `mfem::real_t` and the aliases in `util/typeDef.h` for core numerics so
  both MFEM single- and double-precision builds remain possible. Use fixed-size
  Eigen types for 2D/3D tensors where dimensions are known.
- Use `MFEM_VERIFY` for always-on runtime/input invariants with actionable
  messages, `MFEM_ASSERT` for internal programmer preconditions, and
  `MFEM_ABORT` only for truly unsupported paths. Check finite values,
  orientation (`det(F) > 0` where required), dimensions, and physical parameter
  domains before state is changed.
- Current Eigen maps of MFEM element vectors assume component-major
  `mfem::Ordering::byVDIM`: a vector field is mapped as `dof x dim`, with one
  column per component. If supporting `byNODES`, translate explicitly and test
  both layouts; never silently reinterpret memory.
- Set the MFEM integration point before evaluating transformations or
  coefficients. Honor an integrator's `IntRule` override, use an adequate rule
  for every nonlinear product, use the same rule in residual and Jacobian
  assembly, and apply `ip.weight` and the transformation measure exactly once.
  For nonpolynomial laws, check sensitivity to increasing quadrature order.
- MFEM coefficients, materials, stress-free deformation objects, and point
  storage are commonly held through non-owning references/pointers. Document
  and preserve lifetime requirements. `ElasticMaterial::at()` and
  `setDeformationGradient()` retain borrowed addresses for the current material
  point; finish stress/tangent evaluation synchronously before their values or
  owners change. Prefer RAII for new ownership; do not copy legacy example
  `new`/`delete` patterns without need.
- Treat pointers passed to MFEM `Add*Integrator` methods as ownership transfers;
  `examples/beamParallel/main.cpp` documents this explicitly for
  `ParNonlinearForm`. The form may own the integrator, while the integrator still
  refers non-owningly to its material, coefficients, and point storage. Arrange
  for the form and its integrator to be destroyed before those borrowed
  dependencies, and never register a short-lived stack integrator with an owning
  form.
- `IntegrationPointStorage` caches reference gradients and compile-time typed
  state by element/face. Each initialized entity is bound to the exact
  finite-element and integration-rule objects used initially. Keep custom rules
  alive, and call `Reset` after reference geometry, topology, finite-element,
  partition, or rule changes. For history-dependent physics, transfer accepted
  state before reset; if no conservative/admissible AMR transfer exists, disable
  AMR for that model rather than silently discarding or reinitializing history.
- Existing material/integrator objects contain mutable caches and reusable
  scratch, and `IntegrationPointStorage` tracks mutable current element/face
  cursors. They are not documented as reentrant or thread-safe. Parallel
  assembly requires per-thread instances/state or an explicit redesign, not
  merely an OpenMP loop around existing objects.

### Tensor and kinematic invariants

- The constitutive representation is three-dimensional even in 2D. Current 2D
  mechanics is plane strain unless a feature explicitly implements and tests a
  different reduction.
- `util::Voigt` orders components as `[xx, yy, zz, xy, yz, xz]`. Strain vectors
  use engineering shear `[eps_xx, eps_yy, eps_zz, 2 eps_xy, 2 eps_yz,
  2 eps_xz]`; stress vectors use unscaled shear. A 6-by-6 tangent must map that
  strain convention to that stress convention. Do not change this contract
  without a repository-wide migration and tests.
- `ElasticMaterial` currently supplies second Piola--Kirchhoff stress and a
  reference-material tangent; Cauchy stress is obtained by push-forward.
  Small strain uses the symmetric displacement gradient. Finite strain uses
  `F = I + grad(u)` and Green--Lagrange strain.
- Local cohesive separations are `[t, n]` in 2D and `[t1, t2, n]` in 3D. The
  irreversible exponential law intentionally excludes compression from normal
  cohesive traction/history; contact resistance must be a separate, documented
  formulation.

## Extending the architecture

### Constitutive models

- Put a reusable constitutive law in `material/`; keep spatial dependence in
  `mfem::Coefficient` inputs. Implement stress and its algorithmically
  consistent reference tangent together. Set the kinematic mode deliberately,
  and return `true` from `SupportsMechanicalStrainInput()` only for a genuine
  additive strain formulation.
- Do not embed thermal/eigenstrain kinematics in an elastic law. Compose them
  through `StressFreeDeformation` so assembly and `StressCoefficient`
  postprocessing follow the same small-strain additive or finite-strain
  multiplicative path.
- Plasticity, viscoplasticity, damage, and other path-dependent laws need
  quadrature-point trial and committed state. Do not hide history in one shared
  `ElasticMaterial` instance. Store typed point state through
  `IntegrationPointStorage` (or a deliberate replacement), and specify
  initialization, update, commit, rollback, AMR transfer, and restart
  behavior. A typed material-state bundle supports one owner of each material
  type. Use separate storage objects when independent instances of the same
  stateful model are required.

### Integrators, solvers, contact, and coupled fields

- Put weak-form assembly and MFEM adapters in `fem/`. Residual and Jacobian
  assembly must use the same kinematics, quadrature, measure, parameters, and
  state snapshot. Keep postprocessing on that same constitutive path.
- A nonlinear residual or line search may evaluate a trial point repeatedly.
  Never commit state during `AssembleElementVector`, `AssembleElementGrad`, or
  face assembly. Trial updates must be deterministic functions of committed
  state and the current trial fields, so residual/Jacobian call order does not
  change the result. Integrate stateful components with the
  `BeginStep`/`CommitStep`/`RollbackStep` lifecycle in
  `StepAwareNonlinearFormIntegrator` or its block counterpart, including nested
  solves and rejected increments. `CZMHistory` and its tests are the example.
- A rejected increment must restore the last accepted unknowns, load factor,
  and every quadrature state. Check finite residuals and solver convergence
  before advancing output or AMR. New nonlinear/continuation drivers that do not
  use the existing solver wrappers must invoke the same lifecycle on success,
  failure, early return, and exception paths.
- Implement contact as a face formulation separate from cohesive opening.
  State the slave/master or symmetric convention, gap and normal orientation,
  active-set/regularization policy, friction history, consistent tangent,
  parallel shared-face ownership, and interaction with CZM before coding.
- Use `mfem::BlockNonlinearForm` for monolithic coupled fields. Define field and
  block ordering and assemble every required off-diagonal derivative. For a
  staggered scheme, define inner/outer convergence and when each field's state
  is accepted. The current thermal expansion path is one-way coupling; a
  monolithic temperature solve also needs displacement--temperature tangent
  blocks. Uncoupled diffusion may remain a scalar MFEM form.
- Add new sources to the owning CMake target and expose a header through
  `Plugin.h` only when it is intended as public repository API. Gate optional
  solver code exactly as MFEM reports capabilities.

## Mathematical implementation standard

Every new or materially changed mathematical model must include nearby code
comments and, when the derivation is more than a few lines, a focused document
under `docs/`. Before implementation or review, record:

1. **Model scope:** small/finite deformation, reference/current configuration,
   dimensional reduction, isotropy/anisotropy, rate and temperature
   assumptions, admissible parameter ranges, and known singular limits.
2. **Notation and units:** define every field, tensor, index, sign/normal
   convention, stress and strain measure, Voigt map, history variable, and the
   dimensions/units of inputs and outputs. The repository has no global unit
   system; each problem must use one internally consistent system.
3. **Discrete equations:** show the energy or strong law when applicable, the
   weak residual, quadrature form, state-update algorithm, and the exact
   linearization. Include material, geometric, interface, and cross-field terms
   needed by the chosen formulation. State whether a tangent is symmetric and
   why; do not force symmetry onto a genuinely nonsymmetric branch.
4. **Provenance:** cite authoritative sources with authors, title,
   venue/publisher and year, exact equation/section/page used, and a DOI or
   stable URL when available. Mark formulas adopted verbatim, formulas adapted
   to this repository's notation, and original derivation steps separately.
   Verify references from the source; never invent metadata, quotations, or
   equation numbers. If a source cannot be verified, say so and do not present
   the citation as evidence.
5. **Verification:** add tests that would fail for a sign, factor-of-two,
   ordering, unit, or state-update error. A citation is not a substitute for a
   tangent check, limiting-case test, or convergence evidence.

For constitutive tangents and coupled Jacobian blocks, compare the implemented
linearization with a centered finite-difference directional derivative and,
where useful, autodiff. Test away from nonsmooth branch points and test branch
transitions separately. Use scaled absolute-plus-relative tolerances based on
`mfem::real_t`.

### Numerical robustness and failure handling

- Validate before inversion, logarithms, normalization, or state updates. This
  includes finite and orientation-preserving deformation gradients, admissible
  elastic parameters, positive cohesive/phase-field length scales, and valid
  time or load increments. Near-incompressibility, zero opening, repeated
  eigenvalues, and vanishing increments are model limits to handle deliberately,
  not reasons to accept NaN or infinity.
- Do not silently clamp an inadmissible determinant, parameter, discriminant, or
  history variable merely to keep a solve running. If a regularized branch is
  part of the model, document its equations and differentiable tangent and test
  the limit as regularization vanishes. Otherwise report the failure without
  modifying committed state so the increment can be rejected.
- Penalties, residual stiffness, damping, and branch tolerances have physical or
  algorithmic consequences. State their units and scaling with mesh size,
  polynomial order, and time/load step where applicable; avoid unexplained magic
  constants. Returning a zero response for invalid input is acceptable only as
  an explicit, tested API contract that leaves trial and committed history
  unchanged.

## Performance standard

- Measure before optimizing. Use an optimized `release` build and record the
  problem/mesh, order, assembly mode, solver tolerances, ranks/threads/device,
  compiler/MFEM configuration, iterations, wall time, and peak memory. Compare
  numerical outputs as well as speed.
- Optimize algorithms and data movement before syntax. Preserve sparse/block
  structure, avoid dense global operators, choose solvers/preconditioners that
  match the actual symmetry and conditioning, and track nonlinear and linear
  iteration regressions. Consider MFEM partial assembly or a different assembly
  level only with an explicit compatible kernel and measured benefit.
- Quadrature and element/face loops are hot. Avoid preventable allocation,
  resizing, virtual setup, coefficient reevaluation, matrix inversion, and
  temporary/copy creation inside them. Pre-size reusable or thread-local
  scratch, prefer fixed-size tensors, pass output buffers/views, use
  `Eigen::Map` and `noalias()` safely, and cache only data whose invalidation is
  well defined. Do not trade away clarity of the governing equations for an
  unmeasured micro-optimization.
- Autodiff is valuable as an implementation route and correctness oracle, but
  benchmark it before placing dynamic AD work in a dominant quadrature path.
  An optimized analytic path should remain checked against AD or finite
  differences.
- Current Eigen/raw-`Data()` assembly is host-only; the repository has no MFEM
  device kernel path. Do not claim GPU support for it. A new device path must
  use MFEM's `Memory` access (`Read`, `Write`, or `ReadWrite`) and MFEM device
  loop conventions consistently, avoid implicit host/device synchronization,
  avoid capturing host-only Eigen/virtual/coefficient objects in kernels, and
  retain a verified host fallback.
- For MPI changes, keep local versus true-DOF ownership explicit, avoid global
  gathers, make reductions collective and output rank-aware, and test one and
  multiple ranks. For threading, eliminate races in material scratch and
  quadrature history before measuring scalability.
- There is no benchmark harness or fixed regression threshold. For a claimed
  optimization, provide reproducible before/after measurements on at least one
  representative case; include a second size or rank count when claiming
  scaling. Do not claim a speedup from Debug timings.

## Correctness and verification checklist

- **Constitutive law:** test the undeformed state, canonical uniaxial/shear/
  volumetric states, 2D/3D behavior where supported, parameter bounds, finite
  outputs, frame indifference for finite-strain laws, energy/stress relation,
  and tangent directional derivatives.
- **History/rate law:** test initialization, monotone loading, unload/reload,
  reversal, time/load-increment dependence, irreversibility or dissipation,
  failed-step rollback, accepted-step commit, nested solve behavior, and state
  transfer if the mesh changes.
- **Integrator/coupling:** test element residuals and every Jacobian block by
  directional difference; add patch/rigid-motion tests, conservation or energy
  balances, and a manufactured/convergence test when introducing a PDE.
- **Solver/parallel path:** verify convergence status and residual, not merely
  process exit; compare partition-independent quantities and run a small MPI
  case when parallel assembly, true-DOF handling, or reductions change.
- Keep tests deterministic, small, and self-contained. Scale tolerances to the
  quantity under test and distinguish single/double precision. Examples and
  visual inspection supplement tests but do not replace them.
- Run the narrowest relevant build/test while iterating, then the full Debug
  CTest suite. Run Release tests and a representative example/benchmark for
  performance, solver, optimization, or floating-point-sensitive changes.

## Safe workflow and definition of done

- Inspect `git status` before and after work. Preserve unrelated user changes;
  never reset, clean, overwrite, or reformat files outside the requested scope.
- Do not edit generated build products. Do not change dependency pins, MFEM
  capability gates, tracked meshes/solver option files, third-party code,
  public APIs, or established physical defaults incidentally. Explain and test
  any intentional change to them.
- Do not silently relax tolerances, suppress nonconvergence, replace a
  consistent tangent with a secant, or add stabilization/damping to make one
  example pass. Document the numerical method and expose consequential choices.
- Review `git diff --check`, the complete diff, and final status. Remove
  accidental output files and report exact commands, configurations, and any
  skipped capability-dependent checks.

A change is done when its scope is contained; math, conventions, ownership, and
limitations are documented; authoritative references are verifiable; residual
and tangent/state behavior have focused regression tests; touched C++ is
formatted; relevant Debug tests pass; performance-sensitive work has Release
evidence without correctness or scaling regressions; and the final diff contains
only intended source/documentation changes.

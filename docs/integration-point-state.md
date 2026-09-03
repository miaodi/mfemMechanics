# Integration-point state

## Design

`IntegrationPointStorage<ElementState, FaceState>` defines the state layout for
one analysis at compile time. Element and interior-face integration points store
their geometry and model state in the same record. `NoIntegrationPointState` is
the default for either domain and has no storage overhead beyond the geometry
record.

The non-templated `IntegrationPointStorageBase` exposes geometry only. Stateless
integrators use this interface, while stateful integrators retain the concrete
storage type needed to access state directly. Point-state access has no run-time
map lookup or RTTI cast.

Storage owns the integration-point records. Materials, coefficients,
integrators, finite elements, integration rules, and the mesh remain borrowed
objects with the lifetime requirements documented by their existing APIs.

## Material traits and bundles

`MaterialPointTraits<Material>` associates a material model with its persistent
point-state type. The default is `NoIntegrationPointState`. A stateful material
specializes the trait:

```cpp
struct PlasticityHistory
{
    mfem::real_t equivalentPlasticStrain{ 0. };
};

template <>
struct plugin::MaterialPointTraits<MyPlasticityMaterial>
{
    using State = PlasticityHistory;
};
```

Distinct material types can then be composed into one element-state schema:

```cpp
using ElementState = plugin::MaterialPointStateBundle<
    PhaseFieldElasticMaterial,
    MyPlasticityMaterial>;

plugin::IntegrationPointStorage<ElementState> pointStorage( &mesh );
```

`MaterialPointStateBundle` uses each material type to select its state, so every
material type may appear at most once in a bundle. Independent instances of the
same stateful model should use separate storage objects.

Irreversible cohesive laws use their dedicated face-state specialization:

```cpp
plugin::CZMHistoryPointStorage pointStorage( &mesh );
plugin::ExponentialCZMIntegrator integrator(
    pointStorage, sigmaMax, tauMax, deltaN, deltaT );
```

Use another `CZMHistoryPointStorage` when two cohesive integrators require
independent histories.

## Migration from `AnyMap`

This is an intentional source-level API change. Existing callers should make
the state schema explicit rather than storing values by string key:

| Previous API | Typed API |
| --- | --- |
| `IntegrationPointStorage` in a type declaration | `IntegrationPointStorage<>` for geometry only, or a specialization with state |
| `GetBodyPointData(i)` | `GetElementPoint(i).State` |
| `GetFacePointData(i)` | `GetFacePoint(i).State` |
| `VisitFacePointData(visitor)` | `VisitFaceStates(visitor)` |
| `AnyMap::get_val<T>(key)` | `state.Get<Material>()` |

Local variable declarations may omit `<>` when class template argument
deduction can infer the default geometry-only specialization from the mesh
pointer.

## State lifecycle

Residual and Jacobian assembly may evaluate the same trial fields repeatedly or
visit rejected Newton iterates. Assembly therefore updates trial state only.
Trial updates must be deterministic functions of the committed state and the
current fields.

The nonlinear-solver lifecycle has the following contract:

| Operation | State action |
| --- | --- |
| `BeginStep` | Initialize trial state from committed state. |
| `CommitStep` | Accept trial state as the new committed state. |
| `RollbackStep` | Discard trial state and restore committed state. |

Nested solver calls commit or roll back point state only when the outermost step
finishes. A nested rollback poisons its outer transaction; the outer driver must
also roll back and must not advance the accepted solution. Phase-field history uses
$H_{\mathrm{trial}}=\max(H_{\mathrm{committed}},\psi^+_{\mathrm{current}})$;
it does not accumulate maxima over Newton iterates that may later be rejected.

All lifecycle callbacks are `noexcept`. `CanCommitStep()` provides the
nonmutating first phase of a transaction: the solver preflights every integrator
before invoking any `CommitStep()`. This prevents a rejected integrator or a C++
exception from leaving only part of a multi-integrator transaction committed.
Errors inside a lifecycle callback are programming-contract violations and
terminate rather than returning with partially updated state.

This strengthens the source-level interface for custom step-aware integrators:
overrides of `BeginStep`, `CommitStep`, and `RollbackStep` must also declare
`noexcept`, and `NonlinearStepContext::CommitStep()` now returns whether the
transaction was accepted. Custom `IntegrationPointStorageBase` implementations
may override `GetMesh()` to participate in output-mesh compatibility checks;
its default implementation returns null.

The solver may retry a rejected increment with a smaller step size, but it does
not return to an older accepted solution. Integration-point histories therefore
retain only committed and trial state, not a sequence of accepted states.

`MultiNewtonAdaptive` calls its trial-state function before every Newton attempt,
including retries after a cutback. The callback argument is global pseudo-time:
a dimensionless continuation coordinate, not physical time. Historical solver
APIs call this coordinate `lambda`; `GetCurrentPseudoTime()` and
`GetPseudoTimeIncrement()` make the intended meaning explicit in new code.
Adaptive continuation requires `iterative_mode=true` so Newton preserves the
trial boundary values and starts from the last accepted solution.

## Invalidation and limitations

An initialized element or face is bound to the exact finite-element and
integration-rule objects used to build its cached geometry. `Reset` destroys
both geometry and state. Call it after changes to reference geometry, topology,
finite elements, partitioning, or integration rules.

There is currently no general AMR transfer or restart serialization for typed
point state. A path-dependent analysis must transfer accepted state before
`Reset`; otherwise AMR must remain disabled for that model.

The storage has mutable current-element and current-face cursors and reusable
build scratch. It is not reentrant or thread-safe. The records use host-side
MFEM and Eigen containers and do not provide a device assembly path.

## Verification

`tests/czm_history_test.cpp` checks empty-state size, element and face state
reuse, reset behavior, material-state composition, typed cohesive history,
deterministic repeated phase-field assembly, nested lifecycle behavior, and
failed-step rollback. `tests/j2_plasticity_test.cpp` checks typed element
history under constitutive and finite-element assembly, transaction-wide
rejection, and restoration of a rejected Newton solution.

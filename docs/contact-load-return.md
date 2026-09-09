# Circle contact: load, return, and residual shape

## Scope and controls

`penalty_contact` retains its defaults: elastic material, monotonic loading,
four increments, and penalty contact. Both `--penalty` and `--semismooth` support:

- `--load-path load|load-return` (default `load`). Return consists of separate
  solves over pseudo-time 0→1 and 1→2, with bottom vertical displacement
  0→maximum→0. `--load-steps` requests increments **per stage**; failed Newton
  trials can cause adaptive subdivision. Pseudo-time is a load parameter, not
  physical time.
- `--material elastic|j2` (default `elastic`). J2 uses the existing
  `J2PlasticityMaterial`, `SolidMechanicsIntegrator`, and typed quadrature storage.
- `--yield-stress` (default 1, positive) and `--hardening-modulus` (default 100,
  nonnegative), in the same stress units as `--youngs-modulus` (default 1000).
  This selects linear **isotropic** hardening; kinematic hardening is not exposed.

This is a serial, small-strain, three-dimensional constitutive law reduced to
**plane strain**, not plane stress or finite-strain plasticity. Contact evaluates
the obstacle gap at displaced coordinates, but the bulk model is appropriate
only for small strains and rotations. The existing
[J2 model](j2-plasticity.md) and [contact formulation](frictionless-penalty-contact.md)
specify the constitutive update, weak residuals, consistent tangents, and source
references; no constitutive or contact law is changed here.

The left boundary keeps `u_x = 0`; the complete bottom keeps prescribed `u_y`.
Returning this prescribed displacement to zero does **not** release the supports
or guarantee zero stress. The obstacle remains fixed. Final nonzero displacement
is a supported residual shape, not the freely relaxed shape of an extracted body.
AMR/history transfer, restart, MPI contact, friction, and large deformation are
not introduced by this example.

The same form, material, and storage survive both solve stages. Accepted Newton
states commit before output; rejected trials roll back through the existing
step lifecycle, including the semismooth operator's nested primal form. The
mixed preconditioner still uses the initial elastic primal matrix; it is not a
plastic-tangent preconditioner and robustness for extreme parameters is unproven.
Existing nonlinear tolerances and iteration limits are unchanged.

## Run and visualize

From the repository root, a tested small yielding case is:

```bash
./build/debug/bin/penalty_contact --penalty --load-path load-return \
  --material j2 --yield-stress 1 --hardening-modulus 100 \
  -nx 2 -ny 2 -steps 8 -output -odir /tmp/opencode/contact-return-penalty

./build/debug/bin/penalty_contact --semismooth --load-path load-return \
  --material j2 --yield-stress 1 --hardening-modulus 100 \
  -nx 2 -ny 2 -steps 8 -output -odir /tmp/opencode/contact-return-mixed
```

The output tests execute these parameters with test-owned output prefixes. Use
a fresh prefix for each run to avoid stale files from earlier runs.
Other parameters retain defaults: unit square, order 1, no refinement,
Poisson ratio 0.3, maximum displacement 0.02, circle center (0.65, 1.26), radius
0.25, initial clearance 0.01, penalty factor 10, contact quadrature order 127,
and 200 independent gap-sampling intervals per face. No randomness, external
mesh, or solver option file is used. All quantities require a consistent unit
system; for example lengths in mm and stresses in MPa.

Open `penalty_contact/penalty_contact.pvd` or
`semismooth_contact/semismooth_contact.pvd` under the chosen prefix. Apply
**Warp By Vector**, select `displacement`, and use scale factor 1 for the physical
shape. The stored vector is `(u_x, u_y, 0)`. Animate from time 0 through peak time
1 to return time 2. Color by `equivalent_plastic_strain` for J2: this is a
reference-volume-weighted, elementwise P0 average of **committed** equivalent
plastic strain, not a quadrature-point maximum or a nodal constitutive value.

Initial and every accepted increment are saved; rejected trials are not.
Open the separate `*_contact_obstacle` collection without warping it; its single
time-zero dataset is fixed visualization geometry. Mixed output additionally
contains the boundary-multiplier collection, also starting at zero. The circle
polyline is for display only; contact uses the analytic circle.

## Verification and observed results

Runs used source base `d70623e` plus this change, existing `build/debug` and
`build/release` configurations, `/usr/bin/c++`, and sibling installed MFEM 4.9.1
in double precision with SuiteSparse. The example ran serially on the host;
UMFPACK factors penalty tangents and the mixed primal preconditioner. The mixed
solve uses GMRES and the existing P0 block preconditioner. No GPU path or speedup
is claimed. Building reconfigured these existing directories without changing
dependency or capability options.

For the 2×2 case above (Debug):

| Method/material | Final displacement infinity norm | Maximum cell-average plastic strain |
| --- | ---: | ---: |
| Penalty / elastic | 6.72e-17 | 0 |
| Penalty / J2, yield 1e6 | 2.17e-18 | 0 |
| Penalty / J2, yield 1 | 6.46818e-3 | 7.58849e-3 |
| Semismooth / elastic | 3.04e-17 | 0 |
| Semismooth / J2, yield 1e6 | 3.47e-18 | 0 |
| Semismooth / J2, yield 1 | 6.56623e-3 | 7.70285e-3 |

All six cases reached the return endpoint with exactly zero driven-boundary
error. Final residual norms were below 2e-14; yielding cases detached from the
obstacle. Peak active contact/downward force is checked at time 1, separately
from final residual, boundary, and mixed multiplier checks. High-yield J2 peak
pressure agrees with elastic within the regression's 0.01% band. These are
coarse-mesh regression results, not mesh- or increment-converged predictions.

Verification commands:

```bash
cmake --build build/debug --parallel 4
ctest --test-dir build/debug --output-on-failure --parallel 4
cmake --build build/release --target penalty_contact --parallel 4
ctest --test-dir build/release \
  -R '(PenaltyContact|SemismoothContact)\.(LoadReturn|OutputVerification|Benchmark)$' \
  --output-on-failure
cmake --build build/release --target j2_plasticity_test --parallel 4
ctest --test-dir build/release -R ContactWithJ2 --output-on-failure
```

`LoadReturn` checks elastic recovery, high-yield equivalence at peak, yielding
residual displacement/plasticity, peak contact, and zero final driven boundary.
`OutputVerification` checks initial/return frames, at least the requested
accepted increments, readable datasets, three-component displacement, plastic
strain, and failures caused by blocked output paths. A combined J2/contact
lifecycle unit test explicitly rejects a larger trial, verifies unchanged
committed plasticity, and reproduces the accepted residual for both methods.
This test exercises rollback forwarding, not a forced adaptive solver failure.

All 181 enabled Debug tests passed (183 registered); the two existing opt-in
contact assembly benchmarks remain disabled. Release's six representative contact
tests and the combined rollback test passed. Single precision, non-SuiteSparse solver fallbacks,
interactive ParaView rendering, and a full Release suite were not verified.

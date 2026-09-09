# Lazy operator sums

[`fem/OperatorSum.h`](../fem/OperatorSum.h) provides two-term linear sums and
scalar scaling in `plugin::operator_algebra`. It is also exposed by `Plugin.h`.
It does not assemble a monolithic matrix, differentiate nonlinear operators,
choose a solver, or provide a general expression-tree framework.

## Usage

Given two persistent linear operators `A` and `B` with matching dimensions:

```cpp
#include "OperatorSum.h"

using plugin::operator_algebra::op;

auto sum = mfem::real_t{ 2.5 } * op(A) + op(B) * mfem::real_t{ -0.5 };
mfem::Vector x(sum->Width()), y(sum->Height());
x = 1.;
sum->Mult(x, y); // y = 2.5*A*x - 0.5*B*x

mfem::Vector test(sum->Height()), transposed(sum->Width());
test = 1.;
sum->MultTranspose(test, transposed); // transposed = 2.5*A^T*test - 0.5*B^T*test
```

`op(A)` returns a small, copyable `OperatorTerm` borrowing `A`. Scaling a term
only computes its coefficient; addition creates a fully initialized owned
operator and scratch vector. No operand is applied during expression creation.
The return type of addition is `std::unique_ptr<mfem::Operator>` because the
application node is selected from the coefficients at construction.

Both `a * op(A)` and `op(A) * a` work, including repeated scaling. Zero and
negative coefficients are allowed. Nonfinite coefficients and overflowing
coefficient products fail explicitly. Near-unit coefficients are never rounded.
The API requires a linear-operator contract; accepting an `mfem::Operator&`
does not establish that an arbitrary user-defined operator is actually linear.

## Unit-scale nodes

The sum factory normalizes exact unit coefficients, including explicitly
supplied runtime values and products of preceding scales:

| Expression | Application path |
| --- | --- |
| `op(A) + op(B)` | Dedicated unscaled node: `A*x + B*x`, with no unit multiplications. |
| `1 * op(A) + op(B) * 1` | The same unscaled node. |
| `a * op(A) + op(B)`, `a != 1` | Left-scaled node: scales only `A*x`. |
| `op(A) + b * op(B)`, `b != 1` | Right-scaled node: scales only `B*x`. |
| `a * op(A) + b * op(B)`, both non-unit | MFEM's weighted `SumOperator`. |

Unscaled and mixed-scale paths are separate `UnitScaleSum` template
instantiations. `if constexpr` selects MFEM's unweighted or singly weighted
vector addition routine for both `Mult` and `MultTranspose`; no unit-factor
test is required to select a sum-node path during each application. Fully
weighted sums delegate to `mfem::SumOperator` rather than duplicating its kernel.
Every path applies the left operand before the right operand and uses one
scratch vector. Rectangular sums resize scratch appropriately for transpose.

For example, `0.5 * (2 * op(A)) + op(B)` takes the unscaled path. A coefficient
of `nextafter(1, 2)` does not. Zero coefficients do not skip operand application;
the helper is not an operand-pruning optimizer.

## Ownership and limits

- The returned pointer owns the sum node and scratch, never the two source
  operators. Both operands must outlive the sum and remain valid for its use.
  The small expression terms themselves may expire after sum construction.
- Direct temporary sources, such as `op(mfem::DenseMatrix(2))`, are rejected
  at compile time. Dereferencing a temporary owner can still yield an lvalue;
  this API cannot extend that object's lifetime. Keep source owners alive.
- Borrowing does not copy matrix entries. Updating an operand in place changes
  what the sum applies. Replacing/destroying an operand invalidates the sum;
  construct a fresh sum from the replacement operator.
- Row and column dimensions must both match. As in MFEM's `SumOperator`, a
  leaf that is an `mfem::Solver` must have `iterative_mode == false`, so it
  does not use the pre-existing output as its initial guess. This operand
  restriction is not a prohibition on iterative solvers applied to the sum.
- Application follows MFEM's output-size and operand aliasing contracts.
  Scratch is mutable; do not apply the same sum concurrently without protection.
- Arbitrary chained sums, operator composition, and transpose-expression
  syntax are outside this API. The resulting operator still supports ordinary
  `MultTranspose`. A single `OperatorTerm` is metadata, not a callable operator.
- This operator-only representation does not add a monolithic CSR conversion
  or a direct-solver path for the complete contact system.

## Contact linearization

`SemismoothRigidContactOperator::GetGradient` assembles the contact blocks and
explicitly requests the primal Jacobian at the current primal state. It then
constructs the top-left block as a complete sum:

```cpp
auto& primalGradient = mPrimalOperator.GetGradient(primal);
mPrimalJacobian =
    mPrimalResidualScale * op(primalGradient) + op(*mDisplacementContactJacobian);
mJacobian->SetBlock(0, 0, mPrimalJacobian.get());
```

The mapped contact contribution already contains its row scale; applying that
scale again would be wrong. The full `BlockOperator` remains the same object,
but its owned top-left sum is replaced per gradient evaluation. It is not set
until the first valid linearization. Public gradient access returns only after
the sum is installed, so no incomplete sum or `SetFirst` operation is needed.

MFEM's primal gradient object can itself be replaced during `GetGradient`;
fresh construction prevents retaining an obsolete operand pointer. A returned
contact Jacobian reference remains valid only until the next contact or
borrowed-primal gradient evaluation, as before. Linear solvers repeatedly apply
the already-linearized blocks; sum `Mult`/`MultTranspose` never call `GetGradient`.

This changes composition and ownership bookkeeping, not the contact equations,
quadrature, stabilization, essential constraints, or nonlinear-solver strategy.
There is a wrapper/scratch allocation per requested gradient, not per quadrature
point or Krylov application. Performance must be measured rather than inferred
from the expression syntax or unit-scale specialization.

## Regression coverage

[`tests/operator_sum_test.cpp`](../tests/operator_sum_test.cpp) checks
rectangular nonsymmetric forward/transpose actions, repeated/zero/negative
scales, exact-unit dispatch versus near-unit values, lazy evaluation, expired
term metadata, retained source ownership, dimensions, solver-operand mode, and
nonfinite/overflow rejection. Compile-time checks cover base/derived const and
nonconst lvalues versus direct temporaries.

`SemismoothContact.RefreshesReplacingPrimalGradientOnlyWhenRequested` in
[`tests/contact_test.cpp`](../tests/contact_test.cpp) uses a nonlinear primal
residual `R_i(y) = y_i + y_i^3` that supplies a distinct Jacobian object on each
request. It checks the two linearizations analytically, verifies row scaling,
and counts gradient calls during repeated forward/transpose applications and
residual evaluations. Existing active contact and mixed-block derivative tests
exercise the composed contact contribution.

## Executed verification

Verified in the existing Debug/Release configurations with GCC 15.2.0 and
double-precision MFEM 4.9.1, on the working tree based on
`bf40b5f6e13152cd47ddf550066e3dd5e909f0f7` including pre-existing changes.
The configurations were regenerated to discover the new test target:

```bash
cmake -S . -B build/debug
cmake -S . -B build/release
cmake --build build/debug --parallel 4
ctest --test-dir build/debug --output-on-failure
cmake --build build/release --target operator_sum_test contact_test penalty_contact --parallel 4
ctest --test-dir build/release \
  -R '^(OperatorSum|RigidObstacle|BoundaryMultiplierSpace|PenaltyContact|SemismoothContact)\.' \
  --output-on-failure
```

All 175 enabled Debug tests and all 60 enabled Release operator/contact/example
checks passed. The two opt-in contact assembly microbenchmarks remain disabled
in CTest; the semismooth one was explicitly run below. Single-precision MFEM,
device execution, and a new distributed contact path were not tested. The
SLEPc-dependent `eigenbuckling` target remains configuration-disabled.

The existing [P0 trace benchmark setup](boundary-multiplier-space.md#executed-verification-2026-09-09)
was used before and after replacing the mutable sum: 3-by-3-by-3 hexahedra,
order-four displacement, default P0 multipliers, unit row scales, and 20
residual/Jacobian assembly pairs per sample after warm-up, with no linear solve.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build/release/bin/contact_test --gtest_also_run_disabled_tests \
  --gtest_filter=SemismoothContact.DISABLED_TraceAssemblyBenchmark
```

The five before samples were 7.89981, 7.95154, 8.01158, 7.68988, and 7.68057
ms/pair (median 7.89981); after samples were 7.73363, 7.79841, 7.82366, 7.84357,
and 7.87912 (median 7.82366). Both printed residual norm 29.7788. These short
end-to-end measurements include sum construction but do not isolate allocation
cost. They show comparable time for this case, not a demonstrated speedup;
CPU affinity, peak memory, and scaling were not measured.

#pragma once

#include <cmath>
#include <memory>
#include <mfem.hpp>

namespace plugin::operator_algebra
{
/** @brief A borrowed linear operator and its scalar coefficient.

    Use op(A) to borrow a persistent operator, then a*op(A) + b*op(B) to
    construct an owned sum operator. Terms only record references and
    coefficients: neither construction nor application requests a gradient.
    The sum owns its scratch but not A or B. Both operands must outlive it,
    retain their dimensions, and remain valid for the requested linearization.
    Direct temporary operands are rejected; callers still manage the lifetime
    of objects reached through pointers or other borrowed references.

    This is a two-term linear sum, not a nonlinear-form or matrix-assembly API.
    Nested expression trees, composition, and transpose expressions are not
    provided. Application follows MFEM's output-size and aliasing contracts;
    mutable scratch makes a sum unsuitable for concurrent applications.
    Exact unit coefficients select unscaled application nodes; near-unit
    coefficients are not rounded. Fully weighted sums use mfem::SumOperator. */
class OperatorTerm
{
private:
    friend OperatorTerm op( const mfem::Operator& operand );
    friend OperatorTerm operator*( mfem::real_t scale, OperatorTerm term );
    friend std::unique_ptr<mfem::Operator> operator+( OperatorTerm left, OperatorTerm right );

    explicit OperatorTerm( const mfem::Operator& operand ) : mOperator( &operand ), mScale( 1. )
    {
    }

    const mfem::Operator* mOperator;
    mfem::real_t mScale;
};

inline OperatorTerm op( const mfem::Operator& operand )
{
    return OperatorTerm( operand );
}

OperatorTerm op( mfem::Operator&& ) = delete;
OperatorTerm op( const mfem::Operator&& ) = delete;

inline OperatorTerm operator*( const mfem::real_t scale, OperatorTerm term )
{
    const mfem::real_t coefficient = scale * term.mScale;
    MFEM_VERIFY( std::isfinite( scale ) && std::isfinite( coefficient ),
                 "Operator scaling requires finite coefficients and a finite product." );
    term.mScale = coefficient;
    return term;
}

inline OperatorTerm operator*( OperatorTerm term, const mfem::real_t scale )
{
    return scale * term;
}

namespace detail
{
// At least one operand has a unit coefficient. Select the vector kernel at
// compile time, so unscaled terms are added without a multiplication by one.
template <bool ScaleFirst, bool ScaleSecond>
class UnitScaleSum final : public mfem::Operator
{
    static_assert( !( ScaleFirst && ScaleSecond ), "Fully weighted sums use mfem::SumOperator." );

public:
    UnitScaleSum( const mfem::Operator& first, const mfem::Operator& second, mfem::real_t scale )
        : mfem::Operator( first.Height(), first.Width() ), mFirst( first ), mSecond( second ), mScale( scale ), mScratch( Height() )
    {
    }

    void Mult( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        Apply<false>( input, output );
    }

    void MultTranspose( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        Apply<true>( input, output );
    }

private:
    template <bool Transpose>
    void Apply( const mfem::Vector& input, mfem::Vector& output ) const
    {
        mScratch.SetSize( Transpose ? Width() : Height() );
        if constexpr ( Transpose )
        {
            mFirst.MultTranspose( input, mScratch );
            mSecond.MultTranspose( input, output );
        }
        else
        {
            mFirst.Mult( input, mScratch );
            mSecond.Mult( input, output );
        }
        if constexpr ( ScaleFirst )
        {
            add( output, mScale, mScratch, output );
        }
        else if constexpr ( ScaleSecond )
        {
            add( mScratch, mScale, output, output );
        }
        else
        {
            add( mScratch, output, output );
        }
    }

    const mfem::Operator& mFirst;
    const mfem::Operator& mSecond;
    mfem::real_t mScale;
    mutable mfem::Vector mScratch;
};
} // namespace detail

inline std::unique_ptr<mfem::Operator> operator+( OperatorTerm left, OperatorTerm right )
{
    // Preserve MFEM SumOperator's input contract for every application path.
    MFEM_VERIFY( left.mOperator->Width() == right.mOperator->Width(), "Operator sum has different widths." );
    MFEM_VERIFY( left.mOperator->Height() == right.mOperator->Height(), "Operator sum has different heights." );
    const auto* leftSolver = dynamic_cast<const mfem::Solver*>( left.mOperator );
    const auto* rightSolver = dynamic_cast<const mfem::Solver*>( right.mOperator );
    MFEM_VERIFY( leftSolver == nullptr || !leftSolver->iterative_mode,
                 "Left sum operand must not be in iterative mode." );
    MFEM_VERIFY( rightSolver == nullptr || !rightSolver->iterative_mode,
                 "Right sum operand must not be in iterative mode." );

    if ( left.mScale == 1. && right.mScale == 1. )
    {
        return std::make_unique<detail::UnitScaleSum<false, false>>( *left.mOperator, *right.mOperator, 1. );
    }
    if ( right.mScale == 1. )
    {
        return std::make_unique<detail::UnitScaleSum<true, false>>( *left.mOperator, *right.mOperator, left.mScale );
    }
    if ( left.mScale == 1. )
    {
        return std::make_unique<detail::UnitScaleSum<false, true>>( *left.mOperator, *right.mOperator, right.mScale );
    }
    return std::make_unique<mfem::SumOperator>( left.mOperator, left.mScale, right.mOperator, right.mScale, false, false );
}
} // namespace plugin::operator_algebra

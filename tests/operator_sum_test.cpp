#include "OperatorSum.h"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mfem.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace
{
using mfem::real_t;
using plugin::operator_algebra::op;

template <typename T, typename = void>
struct CanBorrowOperator : std::false_type
{
};

template <typename T>
struct CanBorrowOperator<T, std::void_t<decltype( op( std::declval<T>() ) )>> : std::true_type
{
};

static_assert( CanBorrowOperator<mfem::Operator&>::value );
static_assert( CanBorrowOperator<const mfem::Operator&>::value );
static_assert( CanBorrowOperator<mfem::DenseMatrix&>::value );
static_assert( CanBorrowOperator<const mfem::DenseMatrix&>::value );
static_assert( !CanBorrowOperator<mfem::Operator&&>::value );
static_assert( !CanBorrowOperator<const mfem::Operator&&>::value );
static_assert( !CanBorrowOperator<mfem::DenseMatrix&&>::value );
static_assert( !CanBorrowOperator<const mfem::DenseMatrix&&>::value );

using Term = decltype( op( std::declval<const mfem::Operator&>() ) );
static_assert( !std::is_default_constructible_v<Term> );
static_assert( !std::is_aggregate_v<Term> );
static_assert( !std::is_constructible_v<Term, const mfem::Operator&> );
static_assert( std::is_copy_constructible_v<Term> );
static_assert( std::is_same_v<decltype( std::declval<Term>() + std::declval<Term>() ), std::unique_ptr<mfem::Operator>> );

constexpr real_t kTolerance = 64 * std::numeric_limits<real_t>::epsilon();

mfem::DenseMatrix MakeMatrix( const std::array<real_t, 6>& entries )
{
    mfem::DenseMatrix matrix( 2, 3 );
    for ( int row = 0; row < 2; ++row )
    {
        for ( int column = 0; column < 3; ++column )
        {
            matrix( row, column ) = entries[3 * row + column];
        }
    }
    return matrix;
}

// Row-major fixtures. The oracle below uses independently calculated entries,
// not either source operator's Mult/MultTranspose or another sum implementation.
const std::array<real_t, 6> kFirstEntries{ 1., -2., 3., 4., 0., -1. };
const std::array<real_t, 6> kSecondEntries{ -3., 5., 2., 1., -4., 6. };

void ExpectActions( const mfem::Operator& sum, const real_t firstScale, const real_t secondScale )
{
    ASSERT_EQ( sum.Height(), 2 );
    ASSERT_EQ( sum.Width(), 3 );
    mfem::Vector input( 3 ), output( 2 );
    input( 0 ) = 2.;
    input( 1 ) = -1.;
    input( 2 ) = 3.;
    output = 17.;
    sum.Mult( input, output );
    const std::array<real_t, 2> expected{ 13 * firstScale - 5 * secondScale, 5 * firstScale + 24 * secondScale };
    for ( int row = 0; row < 2; ++row )
    {
        EXPECT_NEAR( output( row ), expected[row], kTolerance * ( 1 + std::abs( expected[row] ) ) );
    }

    mfem::Vector transposeInput( 2 ), transposeOutput( 3 );
    transposeInput( 0 ) = 3.;
    transposeInput( 1 ) = -2.;
    transposeOutput = -19.;
    sum.MultTranspose( transposeInput, transposeOutput );
    const std::array<real_t, 3> transposeExpected{
        -5 * firstScale - 11 * secondScale, -6 * firstScale + 23 * secondScale, 11 * firstScale - 6 * secondScale };
    for ( int column = 0; column < 3; ++column )
    {
        EXPECT_NEAR( transposeOutput( column ), transposeExpected[column],
                     kTolerance * ( 1 + std::abs( transposeExpected[column] ) ) );
    }
}

std::unique_ptr<mfem::Operator> SumFromLocalTerms( const mfem::Operator& first, const mfem::Operator& second )
{
    const auto firstTerm = op( first );
    const auto copiedTerm = firstTerm;
    const auto secondTerm = op( second ) * real_t{ -.5 };
    return real_t{ 2. } * copiedTerm + secondTerm;
}

struct OperatorCounts
{
    int Mult = 0;
    int MultTranspose = 0;
    int GetGradient = 0;
    int Destructions = 0;
};

class CountingMatrixOperator final : public mfem::Operator
{
public:
    CountingMatrixOperator( const mfem::DenseMatrix& matrix, OperatorCounts& counts )
        : mfem::Operator( matrix.Height(), matrix.Width() ), mMatrix( matrix ), mCounts( counts )
    {
    }

    ~CountingMatrixOperator() override
    {
        ++mCounts.Destructions;
    }

    void Mult( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        ++mCounts.Mult;
        mMatrix.Mult( input, output );
    }

    void MultTranspose( const mfem::Vector& input, mfem::Vector& output ) const override
    {
        ++mCounts.MultTranspose;
        mMatrix.MultTranspose( input, output );
    }

    mfem::Operator& GetGradient( const mfem::Vector& ) const override
    {
        ++mCounts.GetGradient;
        return mMatrix;
    }

private:
    mutable mfem::DenseMatrix mMatrix;
    OperatorCounts& mCounts;
};
} // namespace

TEST( OperatorSum, BareRectangularSumHasCorrectForwardAndTransposeActions )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    const auto sum = op( first ) + op( second );
    ASSERT_NE( sum, nullptr );
    ExpectActions( *sum, 1., 1. );
}

TEST( OperatorSum, ScalingWorksOnBothSidesAndPreservesCopiedTerms )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    const auto firstTerm = op( first );
    const auto secondTerm = op( second );
    const auto firstCopy = firstTerm;
    const auto leftRight = real_t{ 2.5 } * firstCopy + secondTerm * real_t{ -.5 };
    const auto rightLeft = firstTerm * real_t{ 2.5 } + real_t{ -.5 } * secondTerm;
    ExpectActions( *leftRight, 2.5, -.5 );
    ExpectActions( *rightLeft, 2.5, -.5 );

    const auto repeated =
        real_t{ -2. } * ( firstTerm * real_t{ .5 } ) * real_t{ 3. } + ( real_t{ 4. } * secondTerm ) * real_t{ -.25 };
    ExpectActions( *repeated, -3., -1. );
    const auto unchanged = firstTerm + secondTerm;
    ExpectActions( *unchanged, 1., 1. );
}

TEST( OperatorSum, ExactUnitCoefficientsSelectTheUnscaledApplicationType )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    real_t one = 1.; // Explicit scalar arguments use the same runtime coefficient API.
    const auto bare = op( first ) + op( second );
    const auto explicitUnit = one * op( first ) + op( second ) * one;
    const auto normalized = real_t{ .5 } * ( real_t{ 2. } * op( first ) ) + ( real_t{ -1. } * op( second ) ) * real_t{ -1. };
    const auto weighted = real_t{ 2. } * op( first ) + real_t{ 3. } * op( second );
    EXPECT_TRUE( typeid( *bare ) == typeid( *explicitUnit ) );
    EXPECT_TRUE( typeid( *bare ) == typeid( *normalized ) );
    EXPECT_FALSE( typeid( *bare ) == typeid( *weighted ) );
    ExpectActions( *bare, 1., 1. );
    ExpectActions( *explicitUnit, 1., 1. );
    ExpectActions( *normalized, 1., 1. );
    ExpectActions( *weighted, 2., 3. );
}

TEST( OperatorSum, MixedUnitCoefficientsSelectTheCorrespondingApplicationTypes )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    real_t one = 1.;
    const auto bare = op( first ) + op( second );
    const auto leftScaled = real_t{ 2. } * op( first ) + op( second );
    const auto leftScaledExplicit = real_t{ 2. } * op( first ) + one * op( second );
    const auto rightScaled = op( first ) + op( second ) * real_t{ 3. };
    const auto rightScaledExplicit = op( first ) * one + op( second ) * real_t{ 3. };
    const auto weighted = real_t{ 2. } * op( first ) + real_t{ 3. } * op( second );
    EXPECT_TRUE( typeid( *leftScaled ) == typeid( *leftScaledExplicit ) );
    EXPECT_TRUE( typeid( *rightScaled ) == typeid( *rightScaledExplicit ) );
    EXPECT_FALSE( typeid( *leftScaled ) == typeid( *rightScaled ) );
    for ( const auto* mixed : { leftScaled.get(), rightScaled.get() } )
    {
        EXPECT_FALSE( typeid( *mixed ) == typeid( *bare ) );
        EXPECT_FALSE( typeid( *mixed ) == typeid( *weighted ) );
    }
    ExpectActions( *leftScaled, 2., 1. );
    ExpectActions( *leftScaledExplicit, 2., 1. );
    ExpectActions( *rightScaled, 1., 3. );
    ExpectActions( *rightScaledExplicit, 1., 3. );
}

TEST( OperatorSum, NearUnitCoefficientsAreNotNormalizedToUnit )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    const real_t nearUnit = std::nextafter( real_t{ 1. }, real_t{ 2. } );
    ASSERT_GT( nearUnit, real_t{ 1. } );
    const auto bare = op( first ) + op( second );
    const auto leftNearUnit = nearUnit * op( first ) + op( second );
    const auto rightNearUnit = op( first ) + op( second ) * nearUnit;
    const auto bothNearUnit = nearUnit * op( first ) + op( second ) * nearUnit;
    const auto leftScaled = real_t{ 2. } * op( first ) + op( second );
    const auto rightScaled = op( first ) + op( second ) * real_t{ 3. };
    const auto weighted = real_t{ 2. } * op( first ) + op( second ) * real_t{ 3. };
    // Numerical tolerances cannot distinguish a one-ULP coefficient change;
    // dispatch must nevertheless use exact equality, not a near-unit tolerance.
    EXPECT_TRUE( typeid( *leftNearUnit ) == typeid( *leftScaled ) );
    EXPECT_TRUE( typeid( *rightNearUnit ) == typeid( *rightScaled ) );
    EXPECT_TRUE( typeid( *bothNearUnit ) == typeid( *weighted ) );
    for ( const auto* sum : { leftNearUnit.get(), rightNearUnit.get(), bothNearUnit.get() } )
    {
        EXPECT_FALSE( typeid( *sum ) == typeid( *bare ) );
    }
    ExpectActions( *leftNearUnit, nearUnit, 1. );
    ExpectActions( *rightNearUnit, 1., nearUnit );
    ExpectActions( *bothNearUnit, nearUnit, nearUnit );
}

TEST( OperatorSum, AllowsZeroAndNegativeCoefficients )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    const auto zeroFirst = real_t{ 0. } * op( first ) + op( second ) * real_t{ -2. };
    const auto zeroSecond = real_t{ -3. } * op( first ) + op( second ) * real_t{ 0. };
    const auto bothZero = op( first ) * real_t{ 0. } + real_t{ 0. } * op( second );
    const auto zeroPlusUnit = real_t{ 0. } * op( first ) + op( second );
    const auto unitPlusZero = op( first ) + op( second ) * real_t{ 0. };
    ExpectActions( *zeroFirst, 0., -2. );
    ExpectActions( *zeroSecond, -3., 0. );
    ExpectActions( *bothZero, 0., 0. );
    ExpectActions( *zeroPlusUnit, 0., 1. );
    ExpectActions( *unitPlusZero, 1., 0. );
}

TEST( OperatorSum, LocalTermsMayExpireBeforeTheSum )
{
    const auto first = MakeMatrix( kFirstEntries );
    const auto second = MakeMatrix( kSecondEntries );
    const auto sum = SumFromLocalTerms( first, second );
    ExpectActions( *sum, 2., -.5 );
}

TEST( OperatorSum, ConstructionIsLazyAndApplicationsDoNotRequestGradients )
{
    for ( const real_t firstScale : { real_t{ 1. }, real_t{ 2. } } )
    {
        for ( const real_t secondScale : { real_t{ 1. }, real_t{ -.5 } } )
        {
            SCOPED_TRACE( ::testing::Message() << "scales=" << firstScale << ',' << secondScale );
            OperatorCounts firstCounts, secondCounts;
            const CountingMatrixOperator first( MakeMatrix( kFirstEntries ), firstCounts );
            const CountingMatrixOperator second( MakeMatrix( kSecondEntries ), secondCounts );
            const auto sum = firstScale * op( first ) + op( second ) * secondScale;
            for ( const auto* counts : { &firstCounts, &secondCounts } )
            {
                EXPECT_EQ( counts->Mult, 0 );
                EXPECT_EQ( counts->MultTranspose, 0 );
                EXPECT_EQ( counts->GetGradient, 0 );
            }
            for ( int application = 1; application <= 3; ++application )
            {
                ExpectActions( *sum, firstScale, secondScale );
                for ( const auto* counts : { &firstCounts, &secondCounts } )
                {
                    EXPECT_EQ( counts->Mult, application );
                    EXPECT_EQ( counts->MultTranspose, application );
                    EXPECT_EQ( counts->GetGradient, 0 );
                }
            }
        }
    }
}

TEST( OperatorSum, DestroyingOwnedWrapperDoesNotDestroyBorrowedLeaves )
{
    OperatorCounts firstCounts, secondCounts;
    {
        const CountingMatrixOperator first( MakeMatrix( kFirstEntries ), firstCounts );
        const CountingMatrixOperator second( MakeMatrix( kSecondEntries ), secondCounts );
        {
            auto sum = SumFromLocalTerms( first, second );
            auto owner = std::move( sum );
            EXPECT_EQ( sum, nullptr );
            ExpectActions( *owner, 2., -.5 );
        }
        EXPECT_EQ( firstCounts.Destructions, 0 );
        EXPECT_EQ( secondCounts.Destructions, 0 );
        const auto anotherSum = op( first ) + op( second );
        ExpectActions( *anotherSum, 1., 1. );
    }
    EXPECT_EQ( firstCounts.Destructions, 1 );
    EXPECT_EQ( secondCounts.Destructions, 1 );
}

TEST( OperatorSum, RejectsDifferentWidths )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    const mfem::DenseMatrix first( 2, 3 ), second( 2, 4 );
    for ( const real_t firstScale : { real_t{ 1. }, real_t{ 2. } } )
    {
        for ( const real_t secondScale : { real_t{ 1. }, real_t{ 3. } } )
        {
            EXPECT_DEATH( (void)( firstScale * op( first ) + secondScale * op( second ) ), "different widths" );
        }
    }
}

TEST( OperatorSum, RejectsDifferentHeights )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    const mfem::DenseMatrix first( 2, 3 ), second( 4, 3 );
    for ( const real_t firstScale : { real_t{ 1. }, real_t{ 2. } } )
    {
        for ( const real_t secondScale : { real_t{ 1. }, real_t{ 3. } } )
        {
            EXPECT_DEATH( (void)( firstScale * op( first ) + secondScale * op( second ) ), "different heights" );
        }
    }
}

TEST( OperatorSum, RejectsIterativeSolverOperandsOnEveryApplicationPath )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    const mfem::IdentityOperator identity( 2 );
    mfem::CGSolver solver;
    solver.SetOperator( identity );
    solver.iterative_mode = true;
    for ( const real_t firstScale : { real_t{ 1. }, real_t{ 2. } } )
    {
        for ( const real_t secondScale : { real_t{ 1. }, real_t{ 3. } } )
        {
            SCOPED_TRACE( ::testing::Message() << "scales=" << firstScale << ',' << secondScale );
            // Construction must reject either borrowed solver leaf without
            // invoking it, for unscaled, both mixed, and fully weighted sums.
            EXPECT_DEATH( (void)( firstScale * op( solver ) + secondScale * op( identity ) ), "iterative mode" );
            EXPECT_DEATH( (void)( firstScale * op( identity ) + secondScale * op( solver ) ), "iterative mode" );
        }
    }
}

TEST( OperatorSum, RejectsNonfiniteScaleOnEitherSide )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    const mfem::DenseMatrix matrix( 2, 3 );
    constexpr real_t infinity = std::numeric_limits<real_t>::infinity();
    constexpr real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    for ( const real_t scale : std::array<real_t, 3>{ nan, infinity, -infinity } )
    {
        EXPECT_DEATH( (void)( scale * op( matrix ) ), "finite" );
        EXPECT_DEATH( (void)( op( matrix ) * scale ), "finite" );
        // A zero coefficient does not exempt subsequent scales from validation.
        EXPECT_DEATH( (void)( scale * ( op( matrix ) * real_t{ 0. } ) ), "finite" );
        EXPECT_DEATH( (void)( ( real_t{ 0. } * op( matrix ) ) * scale ), "finite" );
    }
}

TEST( OperatorSum, RejectsCompoundedCoefficientOverflow )
{
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    const mfem::DenseMatrix matrix( 2, 3 );
    constexpr real_t maximum = std::numeric_limits<real_t>::max();
    const auto large = maximum * op( matrix );
    EXPECT_DEATH( (void)( real_t{ 2. } * large ), "finite" );
    EXPECT_DEATH( (void)( large * real_t{ 2. } ), "finite" );
    EXPECT_DEATH( (void)( real_t{ -2. } * large ), "finite" );
    EXPECT_DEATH( (void)( large * real_t{ -2. } ), "finite" );
}

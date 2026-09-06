#include "util.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

template <typename T>
class EigenDecomp : public testing::Test
{
};

using RealTypes = testing::Types<float, double, long double>;
TYPED_TEST_SUITE( EigenDecomp, RealTypes );

TYPED_TEST( EigenDecomp, ReconstructionAndStrainSplitAcrossScales )
{
    using T = TypeParam;
    using Matrix = Eigen::Matrix<T, 3, 3>;
    using Vector = Eigen::Matrix<T, 3, 1>;
    const T tol = 128 * std::numeric_limits<T>::epsilon();
    // Dense orthogonal basis; do not compare individual eigenvectors, whose
    // signs and basis within a repeated eigenspace are not unique.
    Matrix q;
    q << 1, 2, 2, 2, 1, -2, -2, 2, -1;
    q /= 3;
    const Vector spectra[] = { Vector( -2, 1, 4 ), Vector( -2, 1, 1 ), Vector( -2, -2, 1 ), Vector( 2, 2, 2 ), Vector::Zero() };
    // Normalize before taking norms to avoid underflow/overflow in the checks.
    const T small = std::sqrt( std::numeric_limits<T>::min() );
    const T scales[] = { small, T( 1 ), T( 1 ) / small };
    for ( const auto& spectrum : spectra )
    {
        const Matrix base = q * spectrum.asDiagonal() * q.transpose();
        for ( const T scale : scales )
        {
            SCOPED_TRACE( testing::Message() << "spectrum=" << spectrum.transpose() << ", scale=" << scale );
            const Matrix tensor = scale * base;
            Eigen::SelfAdjointEigenSolver<Matrix> eig( tensor );
            ASSERT_EQ( eig.info(), Eigen::Success );
            ASSERT_TRUE( eig.eigenvalues().allFinite() );
            ASSERT_TRUE( eig.eigenvectors().allFinite() );
            const auto& v = eig.eigenvectors();
            const Vector values = eig.eigenvalues() / scale;
            EXPECT_LE( ( values - spectrum ).norm(), tol * ( 1 + spectrum.norm() ) );
            EXPECT_LE( ( v.transpose() * v - Matrix::Identity() ).norm(), tol );
            EXPECT_LE( ( v * values.asDiagonal() * v.transpose() - base ).norm(), tol * ( 1 + base.norm() ) );
            EXPECT_LE( ( base * v - v * values.asDiagonal() ).norm(), tol * ( 1 + base.norm() ) );

            const auto [positive, negative] = util::StrainSplit( tensor );
            ASSERT_TRUE( positive.allFinite() );
            ASSERT_TRUE( negative.allFinite() );
            const Matrix expectedPositive = q * spectrum.cwiseMax( T( 0 ) ).asDiagonal() * q.transpose();
            const Matrix expectedNegative = q * spectrum.cwiseMin( T( 0 ) ).asDiagonal() * q.transpose();
            EXPECT_LE( ( positive / scale - expectedPositive ).norm(), tol * ( 1 + base.norm() ) );
            EXPECT_LE( ( negative / scale - expectedNegative ).norm(), tol * ( 1 + base.norm() ) );
            EXPECT_LE( ( positive / scale + negative / scale - base ).norm(), tol * ( 1 + base.norm() ) );
        }
    }
}

TYPED_TEST( EigenDecomp, PhysicalShearIsNotHalved )
{
    using T = TypeParam;
    using Matrix = Eigen::Matrix<T, 3, 3>;
    Matrix shear = Matrix::Zero();
    shear( 0, 1 ) = shear( 1, 0 ) = 2;
    Matrix expectedPositive = Matrix::Zero();
    expectedPositive.topLeftCorner( 2, 2 ).setOnes();
    Matrix expectedNegative = -expectedPositive;
    expectedNegative( 0, 1 ) = expectedNegative( 1, 0 ) = 1;
    const auto [positive, negative] = util::StrainSplit( shear );
    const T tol = 128 * std::numeric_limits<T>::epsilon();
    EXPECT_LE( ( positive - expectedPositive ).norm(), tol );
    EXPECT_LE( ( negative - expectedNegative ).norm(), tol );
}

TYPED_TEST( EigenDecomp, RejectsInvalidStrain )
{
    using T = TypeParam;
    using Matrix = Eigen::Matrix<T, 3, 3>;
    GTEST_FLAG_SET( death_test_style, "threadsafe" );
    Matrix tensor = Matrix::Zero();
    tensor( 0, 0 ) = std::numeric_limits<T>::infinity();
    EXPECT_DEATH( (void)util::StrainSplit( tensor ), "finite strain" );
    tensor( 0, 0 ) = std::numeric_limits<T>::quiet_NaN();
    EXPECT_DEATH( (void)util::StrainSplit( tensor ), "finite strain" );
    tensor.setZero();
    const T small = std::numeric_limits<T>::min();
    const T scales[] = { small, T( 1 ), std::numeric_limits<T>::max() / 4 };
    for ( const T scale : scales )
    {
        tensor( 0, 1 ) = scale;
        EXPECT_DEATH( (void)util::StrainSplit( tensor ), "symmetric strain" );
    }
}

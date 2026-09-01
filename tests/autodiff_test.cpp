

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#pragma GCC diagnostic pop
#include "SymmetricEigensolver3x3.hpp"
#include "util.h"
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

using namespace autodiff;

// Demonstrate some basic assertions.
TEST( HelloTest, BasicAssertions )
{
    // Expect two strings not to be equal.
    EXPECT_STRNE( "hello", "world" );
    // Expect equality.
    EXPECT_EQ( 7 * 6, 42 );
}

// compare the difference between using dual as constant and using double as constant
TEST( autodiff, dual_vs_double )
{
    VectorXdual2nd x( 2 );
    x << 1, 2;
    VectorXdual2nd p( 8 );
    p << 8, 7, 6, 5, 4, 3, 2, 1;

    // x: diffX, diffY
    // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
    auto func1 = []( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
    {
        const autodiff::dual2nd& deltaT = p( 0 );
        const autodiff::dual2nd& deltaN = p( 1 );
        const autodiff::dual2nd& phiT = p( 2 );
        const autodiff::dual2nd& phiN = p( 3 );

        Eigen::Map<const VectorXdual2nd> dA1( p.data() + 4, 2 );
        Eigen::Map<const VectorXdual2nd> dA2( p.data() + 6, 2 );
        const autodiff::dual2nd q = phiT / phiN;
        const autodiff::dual2nd r = 0.;

        VectorXdual2nd directionT = dA1 + dA2;
        directionT.normalize();

        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
        const autodiff::dual2nd DeltaT = directionT.dot( x );
        const autodiff::dual2nd DeltaN = directionN.dot( x );

        autodiff::dual2nd res = phiN + phiN * autodiff::detail::exp( -DeltaN / deltaN ) *
                                           ( ( autodiff::dual2nd( 1. ) - r + DeltaN / deltaN ) *
                                                 ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
                                             ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / deltaN ) *
                                                 autodiff::detail::exp( -DeltaT * DeltaT / deltaT / deltaT ) );
        return res;
    };

    // x: diffX, diffY
    // p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
    auto func2 = [&p]( const autodiff::VectorXdual2nd& x )
    {
        Eigen::VectorXd pp = p.template cast<double>();

        const double deltaT = pp[0];
        const double deltaN = pp[1];
        const double phiT = pp[2];
        const double phiN = pp[3];

        Eigen::Vector2d dA1;
        dA1 << pp[4], pp[5];
        Eigen::Vector2d dA2;
        dA2 << pp[6], pp[7];
        const double q = phiT / phiN;
        const double r = 0.;

        VectorXdual2nd directionT = dA1 + dA2;
        directionT.normalize();

        static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
        autodiff::VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
        const autodiff::dual2nd DeltaT = directionT.dot( x );
        const autodiff::dual2nd DeltaN = directionN.dot( x );

        autodiff::dual2nd res = phiN + phiN * autodiff::detail::exp( -DeltaN / deltaN ) *
                                           ( ( 1. - r + DeltaN / deltaN ) * ( 1. - q ) / ( r - 1. ) -
                                             ( q + ( r - q ) / ( r - 1. ) * DeltaN / deltaN ) *
                                                 autodiff::detail::exp( -DeltaT * DeltaT / deltaT / deltaT ) );
        return res;
    };

    autodiff::dual2nd u;
    auto T1 = autodiff::gradient( func1, autodiff::wrt( x ), autodiff::at( x, p ), u );
    auto T2 = autodiff::gradient( func2, autodiff::wrt( x ), autodiff::at( x ), u );

    EXPECT_NEAR( ( T1 - T2 ).norm() / T2.norm(), 0, 1e-13 );
}

// https://github.com/autodiff/autodiff/issues/191
// // compare the difference between using dual as constant and using double as constant
// TEST( autodiff, eigen_value_derivatives_Eigen_VS_CloseForm )
// {
//     // eigenvalue from Eigen
//     auto eigenvalue1 = []( const auto& strainVec )
//     {
//         using T = typename std::decay_t<decltype( strainVec( 0 ) )>;
//         auto strainTensor = util::InverseVoigt( strainVec, true );
//         auto eigenValues = strainTensor.eigenvalues();
//         Eigen::Matrix<T, 3, 1> eval;
//         eval << eigenValues( 0 ).real(), eigenValues( 1 ).real(), eigenValues( 2 ).real();
//         std::sort( std::begin( eval ), std::end( eval ), std::less<T>() );
//         return eval;
//     };

//     // eigenvalue from Closed Form
//     auto eigenvalue2 = []( const auto& strainVec )
//     {
//         using T = typename std::decay_t<decltype( strainVec( 0 ) )>;
//         Eigen::Matrix<T, 3, 1> eval;
//         Eigen::Matrix<T, 3, 3> evec;
//         gte::NISymmetricEigensolver3x3<T> eig;

//         eig( strainVec( 0 ), strainVec( 3 ) / 2, strainVec( 5 ) / 2, strainVec( 1 ), strainVec( 4 ) / 2, strainVec( 2 ),
//              1, eval, evec );

//         return eval;
//     };

//     std::random_device rd;
//     std::mt19937 generator( rd() ); // here you could also set a seed
//     std::uniform_real_distribution<double> distribution( -1.E6, 1.E6 );

//     autodiff::Vector6dual2nd strain{ distribution( generator ), distribution( generator ), distribution( generator ),
//                                      distribution( generator ), distribution( generator ), distribution( generator ) };

//     auto eig1 = eigenvalue1( strain );
//     auto eig2 = eigenvalue2( strain );
//     EXPECT_NEAR( autodiff::detail::val( ( eig1 - eig2 ).norm() / eig2.norm() ), 0, 1e-13 );

//     autodiff::dual2nd pi = 3.141592;

//     autodiff::VectorXdual2nd F; // the output vector F = f(x, p, q) evaluated together with Jacobian below

//     Eigen::MatrixXd Jx1 = autodiff::jacobian( eigenvalue1, wrt( strain ), at( strain ),
//                                               F ); // evaluate the function and the Jacobian matrix J\Lambda = d\Lambda/dE

//     Eigen::MatrixXd Jx2 = autodiff::jacobian( eigenvalue2, wrt( strain ), at( strain ),
//                                               F ); // evaluate the function and the Jacobian matrix  J\Lambda = d\Lambda/dE
//     //   compare jacobian
//     for ( int i = 0; i < 6; i++ )
//     {
//         EXPECT_NEAR( autodiff::detail::val( ( Jx1.col( i ) - Jx2.col( i ) ).norm() / Jx2.col( i ).norm() ), 0, 1e-12 );
//     }
// }

// compare the difference between using dual as constant and using double as constant
TEST( autodiff, eigen_value_derivatives_Iterative_VS_CloseForm )
{
    // eigenvalue from Closed Form
    auto eigenvalue1 = []( const auto& strainVec )
    {
        using T = typename std::decay_t<decltype( strainVec( 0 ) )>;
        Eigen::Matrix<T, 3, 1> eval;
        Eigen::Matrix<T, 3, 3> evec;
        gte::NISymmetricEigensolver3x3<T> eig;

        eig( strainVec( 0 ), strainVec( 3 ) / 2, strainVec( 5 ) / 2, strainVec( 1 ), strainVec( 4 ) / 2, strainVec( 2 ),
             1, eval, evec );

        return eval;
    };

    // eigenvalue from Iterative
    auto eigenvalue2 = []( const auto& strainVec )
    {
        using T = typename std::decay_t<decltype( strainVec( 0 ) )>;
        Eigen::Matrix<T, 3, 1> eval;
        Eigen::Matrix<T, 3, 3> evec;
        gte::SymmetricEigensolver3x3<T> eig;

        eig( strainVec( 0 ), strainVec( 3 ) / 2, strainVec( 5 ) / 2, strainVec( 1 ), strainVec( 4 ) / 2, strainVec( 2 ),
             true, 1, eval, evec );

        return eval;
    };

    std::random_device rd;
    std::mt19937 generator( rd() ); // here you could also set a seed
    std::uniform_real_distribution<double> distribution( -1.E6, 1.E6 );

    autodiff::Vector6dual2nd strain{ distribution( generator ), distribution( generator ), distribution( generator ),
                                     distribution( generator ), distribution( generator ), distribution( generator ) };

    auto eig1 = eigenvalue1( strain );
    auto eig2 = eigenvalue2( strain );
    EXPECT_NEAR( autodiff::detail::val( ( eig1 - eig2 ).norm() / eig2.norm() ), 0, 1e-13 );

    autodiff::dual2nd pi = 3.141592;

    autodiff::VectorXdual2nd F; // the output vector F = f(x, p, q) evaluated together with Jacobian below

    Eigen::MatrixXd Jx1 = autodiff::jacobian( eigenvalue1, wrt( strain ), at( strain ),
                                              F ); // evaluate the function and the Jacobian matrix J\Lambda = d\Lambda/dE

    Eigen::MatrixXd Jx2 = autodiff::jacobian( eigenvalue2, wrt( strain ), at( strain ),
                                              F ); // evaluate the function and the Jacobian matrix  J\Lambda = d\Lambda/dE
    //   compare jacobian
    for ( int i = 0; i < 6; i++ )
    {
        EXPECT_NEAR( autodiff::detail::val( ( Jx1.col( i ) - Jx2.col( i ) ).norm() / Jx2.col( i ).norm() ), 0, 1e-13 );
    }
}
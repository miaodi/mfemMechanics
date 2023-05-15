// C++ includes
#include <iostream>

// autodiff include
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <cmath>
#include <unsupported/Eigen/KroneckerProduct>
using namespace autodiff;

// x: u1x, u1y, u2x, u2y, du1x, du1y, du2x, du2y
// p: deltaT, deltaN, phiT, phiN, dA1x, dA1y, dA2x, dA2y
autodiff::dual2nd f( const autodiff::VectorXdual2nd& x, const autodiff::VectorXdual2nd& p )
{
    Eigen::Map<const VectorXdual2nd> U1( x.data(), 2 );
    Eigen::Map<const VectorXdual2nd> U2( x.data() + 2, 2 );
    Eigen::Map<const VectorXdual2nd> dU1( x.data() + 4, 2 );
    Eigen::Map<const VectorXdual2nd> dU2( x.data() + 6, 2 );

    const autodiff::dual2nd& deltaT = p( 0 );
    const autodiff::dual2nd& deltaN = p( 1 );
    const autodiff::dual2nd& phiT = p( 2 );
    const autodiff::dual2nd& phiN = p( 3 );

    Eigen::Map<const VectorXdual2nd> dA1( p.data() + 4, 2 );
    Eigen::Map<const VectorXdual2nd> dA2( p.data() + 6, 2 );
    const autodiff::dual2nd q = phiT / phiN;
    const autodiff::dual2nd r = 0.;

    VectorXdual2nd diff = U1 - U2;
    VectorXdual2nd directionT = dA1 + dA2 + dU1 + dU2;
    directionT.normalize();

    static Eigen::Rotation2Dd rot( EIGEN_PI / 2 );
    VectorXdual2nd directionN = rot.toRotationMatrix() * directionT;
    const autodiff::dual2nd DeltaT = directionT.dot( diff );
    const autodiff::dual2nd DeltaN = directionN.dot( diff );

    autodiff::dual2nd res = phiN + phiN * autodiff::detail::exp( -DeltaN / deltaN ) *
                                       ( ( autodiff::dual2nd( 1. ) - r + DeltaN / deltaN ) *
                                             ( autodiff::dual2nd( 1. ) - q ) / ( r - autodiff::dual2nd( 1. ) ) -
                                         ( q + ( r - q ) / ( r - autodiff::dual2nd( 1. ) ) * DeltaN / deltaN ) *
                                             autodiff::detail::exp( -DeltaT * DeltaT / deltaT / deltaT ) );
    return res;
}

int main()
{
    VectorXdual2nd x( 8 );
    x << 1, 2, 3, 4, 5, 6, 7, 8;
    VectorXdual2nd p( 8 );
    p << 8, 7, 6, 5, 4, 3, 2, 1;
    autodiff::dual2nd u;
    auto T = autodiff::gradient( f, autodiff::wrt( x ), autodiff::at( x, p ), u );
    std::cout << T.transpose() << std::endl;
    return 0;
}
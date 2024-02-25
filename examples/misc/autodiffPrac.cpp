// C++ includes
#include <iostream>

// autodiff include
#include "Plugin.h"
#include <Eigen/Dense>
#include <SymmetricEigensolver3x3.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <cmath>
#include <iomanip>
#include <unsupported/Eigen/KroneckerProduct>
using namespace autodiff;
int main()
{
    const double lambda = 100;
    const double mu = 50;
    const double k = 0.;
    double phi = .2;
    auto potential_energy = [&]( const auto& strainVec )
    {
        using T = typename std::decay_t<decltype( strainVec( 0 ) )>;

        auto curlyBracPos = []( const T& val ) { return val > 0 ? val : static_cast<T>( 0 ); };
        auto curlyBracNeg = []( const T& val ) { return val < 0 ? val : static_cast<T>( 0 ); };

        Eigen::Matrix<T, 3, 1> eval;
        Eigen::Matrix<T, 3, 3> evec;
        gte::SymmetricEigensolver3x3<T> eig;
        const auto strainTensor = util::InverseVoigt( strainVec, true );

        eig( strainVec( 0 ), strainVec( 3 ) / 2, strainVec( 5 ) / 2, strainVec( 1 ), strainVec( 4 ) / 2, strainVec( 2 ),
             true, 1, eval, evec );
        Eigen::Matrix<T, 3, 3> strainPos;
        Eigen::Matrix<T, 3, 3> strainNeg;
        strainPos.setZero();
        strainNeg.setZero();

        for ( int i = 0; i < 3; i++ )
        {
            if ( eval( i ) < 0 )
                strainNeg += eval( i ) * ( evec.col( i ) * evec.col( i ).transpose() );
            else
                strainPos += eval( i ) * ( evec.col( i ) * evec.col( i ).transpose() );
        }

        const T psiPos = lambda / 2 * autodiff::detail::pow( curlyBracPos( strainTensor.trace() ), 2 ) +
                         mu * ( strainPos * strainPos ).trace();

        const T psiNeg = lambda / 2 * autodiff::detail::pow( curlyBracNeg( strainTensor.trace() ), 2 ) +
                         mu * ( strainNeg * strainNeg ).trace();

        return autodiff::detail::eval( ( 1 - k ) * ( std::pow( 1 - phi, 2 ) + k ) * psiPos + psiNeg );
    };

    std::random_device rd;
    std::mt19937 generator( rd() ); // here you could also set a seed
    std::uniform_real_distribution<double> distribution( -1.E6, 1.E6 );
    autodiff::Vector6dual2nd strain{ distribution( generator ), distribution( generator ), distribution( generator ),
                                     distribution( generator ), distribution( generator ), distribution( generator ) };
    auto psi = potential_energy( strain );

    autodiff::dual2nd u;
    Eigen::VectorXd residual = autodiff::gradient( potential_energy, autodiff::wrt( strain ), autodiff::at( strain ), u );
    autodiff::VectorXdual g;
    Eigen::MatrixXd stiffness = autodiff::hessian( potential_energy, autodiff::wrt( strain ), autodiff::at( strain ), u, g );
    std::cout << residual.transpose() << std::endl;
    std::cout << stiffness.transpose() << std::endl;
    return 0;
}
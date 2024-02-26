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
#include <taco.h>
#include <unsupported/Eigen/KroneckerProduct>
using namespace autodiff;
int main()
{
    const double lambda = 100;
    const double mu = 50;
    const double k = 0.;
    double phi = .2;
    auto strain_split = []( const auto& strainTensor )
    {
        using T = typename std::decay_t<decltype( strainTensor( 0, 0 ) )>;

        Eigen::Matrix<T, 3, 1> eval;
        Eigen::Matrix<T, 3, 3> evec;
        gte::SymmetricEigensolver3x3<T> eig;

        eig( strainTensor( 0, 0 ), strainTensor( 0, 1 ) / 2, strainTensor( 0, 2 ) / 2, strainTensor( 1, 1 ),
             strainTensor( 1, 2 ) / 2, strainTensor( 2, 2 ), true, 1, eval, evec );

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
        return std::tuple<Eigen::Matrix<T, 3, 3>, Eigen::Matrix<T, 3, 3>>{ strainPos, strainNeg };
    };

    auto potential_energy = [&]( const auto& strainVec, const int stat )
    {
        using T = typename std::decay_t<decltype( strainVec( 0 ) )>;

        auto curlyBracPos = []( const T& val ) { return val > 0 ? val : static_cast<T>( 0 ); };
        auto curlyBracNeg = []( const T& val ) { return val < 0 ? val : static_cast<T>( 0 ); };

        const auto strainTensor = util::InverseVoigt( strainVec, true );

        auto [strainPos, strainNeg] = strain_split( strainTensor );
        const T psiPos = lambda / 2 * autodiff::detail::pow( curlyBracPos( strainTensor.trace() ), 2 ) +
                         mu * strainPos.array().square().sum();
        if ( stat == 0 )
            return psiPos;
        const T psiNeg = lambda / 2 * autodiff::detail::pow( curlyBracNeg( strainTensor.trace() ), 2 ) +
                         mu * strainNeg.array().square().sum();
        if ( stat == 1 )
            return psiNeg;
        const T res = ( ( 1 - k ) * std::pow( 1 - phi, 2 ) + k ) * psiPos + psiNeg;
        return res;
    };

    std::random_device rd;
    std::mt19937 generator( rd() ); // here you could also set a seed
    std::uniform_real_distribution<double> distribution( -1, 1. );
    Eigen::Vector6d strainDouble{ distribution( generator ), distribution( generator ), distribution( generator ),
                                  distribution( generator ), distribution( generator ), distribution( generator ) };
    Eigen::Vector6dual2nd strain{ strainDouble( 0 ), strainDouble( 1 ), strainDouble( 2 ),
                                  strainDouble( 3 ), strainDouble( 4 ), strainDouble( 5 ) };
    // auto psi = potential_energy( strain );
    // std::cout << "strain: \n" << strain << std::endl;

    autodiff::dual2nd u;
    Eigen::VectorXd residual = autodiff::gradient( potential_energy, autodiff::wrt( strain ), autodiff::at( strain, 2 ), u );
    autodiff::Vector6dual2nd strainIncre{ strainDouble( 0 ) + 1e-8, strainDouble( 1 ), strainDouble( 2 ),
                                          strainDouble( 3 ),        strainDouble( 4 ), strainDouble( 5 ) };
    std::cout << std::setprecision( 16 )
              << autodiff::detail::val( potential_energy( strainIncre, 2 ) - potential_energy( strain, 2 ) ) / 1e-8
              << std::endl;
    autodiff::VectorXdual g;
    // Eigen::MatrixXd stiffness = autodiff::hessian( potential_energy, autodiff::wrt( strain ), autodiff::at( strain ), u, g );
    std::cout << std::setprecision( 16 ) << residual.transpose() << std::endl;
    // std::cout << stiffness.transpose() << std::endl;

    {
        taco::IndexVar I, J, K, L;
        // taco::IndexVar m, n, o, p;

        taco::Format sd2( { taco::Dense, taco::Dense } );

        // Create tensors
        taco::Tensor<double> I2( { 3, 3 }, sd2 );
        taco::Tensor<double> strainTensor( { 3, 3 }, sd2 );
        taco::Tensor<double> strainPosTensor( { 3, 3 }, sd2 );
        taco::Tensor<double> strainNegTensor( { 3, 3 }, sd2 );
        taco::Tensor<double> stressTensor( { 3, 3 }, sd2 );

        // Insert data identity tensor I2
        I2.insert( { 0, 0 }, 1. );
        I2.insert( { 1, 1 }, 1. );
        I2.insert( { 2, 2 }, 1. );

        // Pack inserted data as described by the formats
        I2.pack();

        auto change_storage = []( Eigen::Matrix3d& eigen, taco::Tensor<double>& tensor )
        {
            auto _array = taco::makeArray<double>( eigen.data(), 9 );
            taco::TensorStorage& _storage = tensor.getStorage();
            _storage.setValues( _array );
            tensor.setStorage( _storage );
        };
        auto strainTensorEigen = util::InverseVoigt( strainDouble, true );
        auto [strainPosTensorEigen, strainNegTensorEigen] = strain_split( strainTensorEigen );

        // Eigen::Matrix3d stressTensorEigen;

        change_storage( strainTensorEigen, strainTensor );
        change_storage( strainPosTensorEigen, strainPosTensor );
        change_storage( strainNegTensorEigen, strainNegTensor );
        const double pos = strainTensorEigen.trace() > 0 ? strainTensorEigen.trace() : 0;
        const double neg = strainTensorEigen.trace() < 0 ? strainTensorEigen.trace() : 0;
        // stressTensor( i, j ) = static_cast<double>( ( 1. - k ) * std::pow( 1. - phi, 2 ) ) *
        //                        ( lambda * pos * I2( i, j ) + 2 * mu * strainPosTensor( i, j ) );
        stressTensor( I, J ) =
            ( ( 1 - k ) * std::pow( 1 - phi, 2 ) + k ) * ( lambda * pos * I2( I, J ) + 2 * mu * strainPosTensor( I, J ) ) +
            lambda * neg * I2( I, J ) + 2 * mu * strainNegTensor( I, J );
        std::cout << std::setprecision( 16 ) << stressTensor( 0, 0 ) << " " << stressTensor( 0, 1 ) << " "
                  << stressTensor( 0, 2 ) << " " << std::endl;
        std::cout << std::setprecision( 16 ) << stressTensor( 1, 0 ) << " " << stressTensor( 1, 1 ) << " "
                  << stressTensor( 1, 2 ) << " " << std::endl;
        std::cout << std::setprecision( 16 ) << stressTensor( 2, 0 ) << " " << stressTensor( 2, 1 ) << " "
                  << stressTensor( 2, 2 ) << " " << std::endl;

        // std::cout << strainPos << std::endl;
    }
    return 0;
}
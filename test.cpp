#include "Plugin.h"
#include "mfem.hpp"
// #include <Fastor/Fastor.h>
#include <fstream>
#include <iostream>
#include <taco.h>
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;
using namespace Eigen;
using namespace mfem;

using short_t = short;
// Indices of the Voigt notation
inline short_t voigt( short_t dim, short_t I, short_t J )
{
    if ( dim == 2 )
        switch ( I )
        {
        case 0:
            return J == 0 ? 0 : 0;
        case 1:
            return J == 0 ? 1 : 1;
        case 2:
            return J == 0 ? 0 : 1;
        }
    else if ( dim == 3 )
        switch ( I )
        {
        case 0:
            return J == 0 ? 0 : 0;
        case 1:
            return J == 0 ? 1 : 1;
        case 2:
            return J == 0 ? 2 : 2;
        case 3:
            return J == 0 ? 0 : 1;
        case 4:
            return J == 0 ? 1 : 2;
        case 5:
            return J == 0 ? 0 : 2;
        }
    return -1;
}
inline void voigtStress( VectorXd& Svec, const MatrixXd& S )
{
    short_t dim = S.cols();
    short_t dimTensor = dim * ( dim + 1 ) / 2;
    Svec.resize( dimTensor );
    for ( short i = 0; i < dimTensor; ++i )
        Svec( i ) = S( voigt( dim, i, 0 ), voigt( dim, i, 1 ) );
}

inline void setB( MatrixXd& B, const MatrixXd& F, const VectorXd& bGrad )
{
    short_t dim = F.cols();
    short_t dimTensor = dim * ( dim + 1 ) / 2;
    B.resize( dimTensor, dim );

    for ( short_t j = 0; j < dim; ++j )
    {
        for ( short_t i = 0; i < dim; ++i )
            B( i, j ) = F( j, i ) * bGrad( i );
        if ( dim == 2 )
            B( 2, j ) = F( j, 0 ) * bGrad( 1 ) + F( j, 1 ) * bGrad( 0 );
        if ( dim == 3 )
            for ( short_t i = 0; i < dim; ++i )
            {
                short_t k = ( i + 1 ) % dim;
                B( i + dim, j ) = F( j, i ) * bGrad( k ) + F( j, k ) * bGrad( i );
            }
    }
}

int main()
{
    // mfem::DenseMatrix dm;
    // dm.SetSize( 5 );
    // dm = 1.0;
    // dm( 1, 3 ) = 2.;
    // dm.Print( cout, 10 );

    // auto ptr = dm.Data();
    // Map<Eigen::Matrix<double, 5, 5>> elmat( ptr );
    // elmat += MatrixXd::Random( 5, 5 );
    // dm.Print( cout, 10 );

    // MatrixXd test = MatrixXd::Random( 3, 3 );
    // MatrixXd test2 = Eigen::kroneckerProduct( test, MatrixXd::Identity( 3, 3 ) );
    // cout << test2 << endl << endl;
    // MatrixXd test3 = Eigen::kroneckerProduct( MatrixXd::Identity( 3, 3 ), test );
    // cout << test3 << endl;

    // Matrix3d F{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
    // MatrixXd B;
    // Vector3d grad{ { 4, 6, 7 } };

    // setB( B, F, grad );
    // cout << B << endl;

    // int dim = 3, dof = 1;
    // MatrixXd B2( 6, dof * dim );
    // for ( int i = 0; i < dof; i++ )
    // {
    //     B2( 0, i + 0 * dof ) = grad( 0 ) * F( 0, 0 );
    //     B2( 0, i + 1 * dof ) = grad( 0 ) * F( 1, 0 );
    //     B2( 0, i + 2 * dof ) = grad( 0 ) * F( 2, 0 );

    //     B2( 1, i + 0 * dof ) = grad( 1 ) * F( 0, 1 );
    //     B2( 1, i + 1 * dof ) = grad( 1 ) * F( 1, 1 );
    //     B2( 1, i + 2 * dof ) = grad( 1 ) * F( 2, 1 );

    //     B2( 2, i + 0 * dof ) = grad( 2 ) * F( 0, 2 );
    //     B2( 2, i + 1 * dof ) = grad( 2 ) * F( 1, 2 );
    //     B2( 2, i + 2 * dof ) = grad( 2 ) * F( 2, 2 );

    //     B2( 3, i + 0 * dof ) = grad( 1 ) * F( 0, 0 ) + grad( 0 ) * F( 0, 1 );
    //     B2( 3, i + 1 * dof ) = grad( 1 ) * F( 1, 0 ) + grad( 0 ) * F( 1, 1 );
    //     B2( 3, i + 2 * dof ) = grad( 1 ) * F( 2, 0 ) + grad( 0 ) * F( 2, 1 );

    //     B2( 5, i + 0 * dof ) = grad( 0 ) * F( 0, 2 ) + grad( 2 ) * F( 0, 0 );
    //     B2( 5, i + 1 * dof ) = grad( 0 ) * F( 1, 2 ) + grad( 2 ) * F( 1, 0 );
    //     B2( 5, i + 2 * dof ) = grad( 0 ) * F( 2, 2 ) + grad( 2 ) * F( 2, 0 );

    //     B2( 4, i + 0 * dof ) = grad( 2 ) * F( 0, 1 ) + grad( 1 ) * F( 0, 2 );
    //     B2( 4, i + 1 * dof ) = grad( 2 ) * F( 1, 1 ) + grad( 1 ) * F( 1, 2 );
    //     B2( 4, i + 2 * dof ) = grad( 2 ) * F( 2, 1 ) + grad( 1 ) * F( 2, 2 );
    // }
    // Rotation2Dd r( EIGEN_PI / 2 );
    // cout << r.toRotationMatrix() << endl;
    {
        // Create formats
        taco::Format sv4( { taco::Dense, taco::Dense, taco::Dense, taco::Dense } );
        taco::Format sv2( { taco::Sparse, taco::Sparse } );
        taco::Format sd2( { taco::Dense, taco::Dense } );

        // Create tensors
        taco::Tensor<double> I2( { 3, 3 }, sv2 );
        taco::Tensor<double> I4( { 3, 3, 3, 3 }, sv4 );
        taco::Tensor<double> C( { 3, 3, 3, 3 }, sv4 );

        // Insert data into B and c
        I2.insert( { 0, 0 }, 1. );
        I2.insert( { 1, 1 }, 1. );
        I2.insert( { 2, 2 }, 1. );

        // Pack inserted data as described by the formats
        I2.pack();

        Matrix3d rand = MatrixXd::Random( 3, 3 );

        taco::Array A2( taco::Float64, rand.data(), 9, taco::Array::UserOwns );

        std::cout << rand << std::endl;

        taco::Tensor<double> x( { 3, 3 }, { taco::Dense, taco::Dense } );
        // Array x_array = makeArray<double>( x_values, 10 );
        taco::TensorStorage x_storage = x.getStorage();
        x_storage.setValues( A2 );
        x.setStorage( x_storage );
        auto T = x.transpose( { 1, 0 } );

        // Lame

        const double lambda = 1;
        const double mu = 0;

        // Form a tensor-vector multiplication expression
        taco::IndexVar i, j, k, l;
        taco::IndexVar m, n, o, p;
        I4( i, j, k, l ) = .5 * ( I2( i, k ) * I2( j, l ) + I2( i, l ) * I2( j, k ) );

        C( i, j, k, l ) = lambda * I2( i, j ) * I2( k, l ) + 2 * mu * I4( i, j, k, l );
        C.evaluate();

        taco::Tensor<double> CT( { 3, 3, 3, 3 }, sv4 );
        CT( i, j, k, l ) = T( m, i ) * T( n, j ) * T( o, k ) * T( p, l ) * C( m, n, o, p );

        Matrix6d EigenC;

        for ( int i = 0; i < 6; i++ )
        {
            for ( int j = 0; j < 6; j++ )
            {
                EigenC( i, j ) = C.at( { (int)util::Voigt( i, 0 ), (int)util::Voigt( i, 1 ), (int)util::Voigt( j, 2 ),
                                         (int)util::Voigt( j, 3 ) } );
            }
        }
        Eigen::Matrix6d Trans = util::TransformationVoigtForm( rand );
        std::cout << Trans.transpose() * EigenC * Trans << std::endl << std::endl;
        Matrix6d EigenCTACO;
        for ( int i = 0; i < 6; i++ )
        {
            for ( int j = 0; j < 6; j++ )
            {
                EigenCTACO( i, j ) = CT.at( { (int)util::Voigt( i, 0 ), (int)util::Voigt( i, 1 ),
                                              (int)util::Voigt( j, 2 ), (int)util::Voigt( j, 3 ) } );
            }
        }

        // MFEM_VERIFY( 1 == 2, "fuck." );
        std::cout << EigenCTACO - Trans.transpose() * EigenC * Trans << std::endl;
    }
    // {
    //     enum
    //     {
    //         i,
    //         j,
    //         k,
    //         l
    //     };

    //     // An example of Einstein summation

    //     Fastor::Tensor<double, 3, 3> I2;
    //     I2.eye();
    //     Fastor::Tensor<double, 3, 3, 3, 3> I4 =
    //         .5 * ( Fastor::permutation<Fastor::Index<i, k, j, l>>( Fastor::outer( I2, I2 ) ) +
    //                Fastor::permutation<Fastor::Index<i, l, j, k>>( Fastor::outer( I2, I2 ) ) );
    //     const double lambda = 1.;
    //     const double mu = .5;
    //     Fastor::Tensor<double, 3, 3, 3, 3> C = lambda * Fastor::outer( I2, I2 ) + 2 * mu * I4;
    //     Matrix6d EigenC;
    //     for ( int i = 0; i < 6; i++ )
    //     {
    //         for ( int j = 0; j < 6; j++ )
    //         {
    //             EigenC( i, j ) = C( (int)util::Voigt( i, 0 ), (int)util::Voigt( i, 1 ), (int)util::Voigt( j, 2 ),
    //                                 (int)util::Voigt( j, 3 ) );
    //         }
    //     }

    //     std::cout << EigenC << std::endl;

    //     Fastor::_voigt<double, 3, 3, 3, 3>( C.data(), EigenC.data() );

    //     std::cout << EigenC << std::endl;
    //     // std::cout << Fastor::permutation<Fastor::Index<i, k, j, l>>( Fastor::outer( I2, I2 ) ) << std::endl;
    //     // std::cout << Fastor::permutation<Fastor::Index<i, l, j, k>>( Fastor::outer( I2, I2 ) ) << std::endl;
    //     // std::cout << Fastor::permute<Fastor::Index<i, j, k, l>>(
    //     //                  Fastor::einsum<Fastor::Index<i, k>, Fastor::Index<j, l>>( I2, I2 ) +
    //     //                  Fastor::einsum<Fastor::Index<i, l>, Fastor::Index<j, k>>( I2, I2 ) )
    //     //           << std::endl;
    // }
    return 0;
}
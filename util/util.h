#pragma once

#include "typeDef.h"
#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <cmath>
#include <iostream>
#include <mfem.hpp>

namespace util
{
constexpr double pi = 3.14159265358979323846;

template <typename... Args>
void mfemOut( Args&&... args )
{
#ifdef MFEM_USE_MPI
    int init_flag, fin_flag;
    MPI_Initialized( &init_flag );
    MPI_Finalized( &fin_flag );
    if ( init_flag && !fin_flag )
    {
        int rank = 0;
        MPI_Comm_rank( mfem::GetGlobalMPI_Comm(), &rank );
        if ( rank == 0 )
        {
            ( mfem::out << ... << args );
        }
    }
    else
    {
        ( mfem::out << ... << args );
    }
#else
    ( mfem::out << ... << args );
#endif
}

Eigen::Vector6d Voigt( const Eigen::Matrix3d& tensor, const bool isStrain );

template <typename T>
Eigen::Matrix<T, 3, 3> InverseVoigt( const Eigen::Vector<T, 6>& vector, const bool isStrain )
{
    if ( isStrain )
    {
        return Eigen::Matrix<T, 3, 3>{ { vector( 0 ), vector( 3 ) / 2, vector( 5 ) / 2 },
                                       { vector( 3 ) / 2, vector( 1 ), vector( 4 ) / 2 },
                                       { vector( 5 ) / 2, vector( 4 ) / 2, vector( 2 ) } };
    }
    else
    {
        return Eigen::Matrix<T, 3, 3>{ { vector( 0 ), vector( 3 ), vector( 5 ) },
                                       { vector( 3 ), vector( 1 ), vector( 4 ) },
                                       { vector( 5 ), vector( 4 ), vector( 2 ) } };
    }
}

short Voigt( const short i, const short pos );

void symmetricIdentityTensor( const Eigen::Matrix3d& C, Eigen::Matrix6d& CC );

void tensorProduct( const Eigen::Matrix3d& A, const Eigen::Matrix3d& B, Eigen::Matrix6d& CC );

Eigen::Matrix6d TransformationVoigtForm( const Eigen::Matrix3d& transformation );

double ConvergenceRate( const double cur, const double prev, const double prevprev );

double SmallestCircle( const mfem::IntegrationRule& nodes, const int dim );

template <typename T>
void EigenDecomp( const Eigen::Matrix<T, 3, 3>& sym3, Eigen::Matrix<T, 3, 1>& eigVal, Eigen::Matrix<T, 3, 3>& eigVec )
{
    const T& a = sym3( 0, 0 );
    const T& b = sym3( 1, 1 );
    const T& c = sym3( 2, 2 );
    const T& d = sym3( 0, 1 );
    const T& e = sym3( 1, 2 );
    const T& f = sym3( 0, 2 );

    const T x1 = autodiff::detail::pow( a, 2 ) + autodiff::detail::pow( b, 2 ) + autodiff::detail::pow( c, 2 ) - a * b -
                 a * c - b * c +
                 3 * ( autodiff::detail::pow( d, 2 ) + autodiff::detail::pow( e, 2 ) + autodiff::detail::pow( f, 2 ) );
    const T x2 = -( 2 * a - b - c ) * ( 2 * b - a - c ) * ( 2 * c - a - b ) +
                 9 * ( ( 2 * c - a - b ) * autodiff::detail::pow( d, 2 ) + ( 2 * b - a - c ) * autodiff::detail::pow( f, 2 ) +
                       ( 2 * a - b - c ) * autodiff::detail::pow( e, 2 ) ) -
                 54 * d * e * f;
    T phi = 0;
    if ( x2 > 0 )
        phi = autodiff::detail::atan(
            autodiff::detail::sqrt( 4 * autodiff::detail::pow( x1, 3 ) - autodiff::detail::pow( x2, 2 ) ) / x2 );
    else if ( x2 == 0 )
        phi = pi / 2;
    else
        phi = autodiff::detail::atan(
                  autodiff::detail::sqrt( 4 * autodiff::detail::pow( x1, 3 ) - autodiff::detail::pow( x2, 2 ) ) / x2 ) +
              pi;

    eigVal( 0 ) = ( a + b + c - 2 * sqrt( x1 ) * autodiff::detail::cos( phi / 3 ) ) / 3;
    eigVal( 1 ) = ( a + b + c + 2 * sqrt( x1 ) * autodiff::detail::cos( ( phi - pi ) / 3 ) ) / 3;
    eigVal( 2 ) = ( a + b + c + 2 * sqrt( x1 ) * autodiff::detail::cos( ( phi + pi ) / 3 ) ) / 3;

    for ( int i = 0; i < 3; i++ )
    {
        const T m = ( d * ( c - eigVal( i ) ) - e * f ) / ( f * ( b - eigVal( i ) - d * e ) );

        eigVec( 0, i ) = ( eigVal( i ) - c - e * m ) / f;
        eigVec( 1, i ) = m;
        eigVec( 2, i ) = 1;
    }
}
} // namespace util
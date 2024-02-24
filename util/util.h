#pragma once

#include "typeDef.h"
#include <Eigen/Dense>
#include <iostream>
#include <mfem.hpp>

namespace util
{
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
} // namespace util
#pragma once

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
} // namespace util
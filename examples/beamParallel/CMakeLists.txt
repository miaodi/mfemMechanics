if(MFEM_USE_MPI AND MFEM_USE_MUMPS)
    mfem_mechanics_add_executable(
        beamParallel
        main.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping beamParallel: requires MFEM_USE_MPI=ON and MFEM_USE_MUMPS=ON "
        "(found MPI=${MFEM_USE_MPI}, MUMPS=${MFEM_USE_MUMPS})"
    )
endif()

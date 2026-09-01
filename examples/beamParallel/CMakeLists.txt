if(MFEM_USE_MPI AND MFEM_USE_PETSC)
    mfem_mechanics_add_executable(
        beamParallel
        main.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping beamParallel: requires MFEM_USE_MPI=ON and MFEM_USE_PETSC=ON "
        "(found MPI=${MFEM_USE_MPI}, PETSc=${MFEM_USE_PETSC})"
    )
endif()

if(MFEM_USE_MPI AND MFEM_USE_PETSC)
    mfem_mechanics_add_executable(
        postBuckling3D1
        main1.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
    mfem_mechanics_add_executable(
        postBuckling3D2
        main2.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping postBuckling3D1 and postBuckling3D2: requires MFEM_USE_MPI=ON and "
        "MFEM_USE_PETSC=ON (found MPI=${MFEM_USE_MPI}, PETSc=${MFEM_USE_PETSC})"
    )
endif()

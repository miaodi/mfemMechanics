if(MFEM_USE_MPI AND MFEM_USE_PETSC AND MFEM_USE_SLEPC)
    mfem_mechanics_add_executable(
        eigenbuckling
        main.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping eigenbuckling: requires MFEM_USE_MPI=ON, MFEM_USE_PETSC=ON, and "
        "MFEM_USE_SLEPC=ON (found MPI=${MFEM_USE_MPI}, PETSc=${MFEM_USE_PETSC}, "
        "SLEPc=${MFEM_USE_SLEPC})"
    )
endif()

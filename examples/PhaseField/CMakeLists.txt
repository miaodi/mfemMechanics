if(TARGET OpenMP::OpenMP_CXX AND MFEM_USE_SUITESPARSE)
    mfem_mechanics_add_executable(
        PhaseField_shear
        shear.cpp
        mfemMechanics::femplugin
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS
        "Skipping PhaseField_shear: requires OpenMP CXX and MFEM_USE_SUITESPARSE=ON "
        "(found OpenMP=${OpenMP_CXX_FOUND}, SuiteSparse=${MFEM_USE_SUITESPARSE})"
    )
endif()

if(MFEM_USE_MPI)
    mfem_mechanics_add_executable(
        pPhaseField_shear
        pshear.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS "Skipping pPhaseField_shear: installed MFEM reports MFEM_USE_MPI=OFF")
endif()

if(TARGET OpenMP::OpenMP_CXX)
    mfem_mechanics_add_executable(
        PhaseField_shear
        shear.cpp
        mfemMechanics::femplugin
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "Skipping PhaseField_shear: OpenMP CXX support was not found")
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

if(TARGET OpenMP::OpenMP_CXX)
    mfem_mechanics_add_executable(
        czm
        main.cpp
        mfemMechanics::femplugin
        OpenMP::OpenMP_CXX
    )
    mfem_mechanics_add_executable(
        czm2
        main2.cpp
        mfemMechanics::femplugin
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "Skipping czm and czm2: OpenMP CXX support was not found")
endif()

if(MFEM_USE_MPI AND MFEM_USE_PETSC)
    mfem_mechanics_add_executable(
        czm2p
        main2p.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping czm2p: requires MFEM_USE_MPI=ON and MFEM_USE_PETSC=ON "
        "(found MPI=${MFEM_USE_MPI}, PETSc=${MFEM_USE_PETSC})"
    )
endif()

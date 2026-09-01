mfem_mechanics_add_executable(heat_dynamic dynamic.cpp mfem)
mfem_mechanics_add_executable(heat_static static.cpp mfem)

if(MFEM_USE_MPI AND MFEM_USE_PETSC)
    mfem_mechanics_add_executable(
        thermalStrain
        thermalStrain.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping thermalStrain: requires MFEM_USE_MPI=ON and MFEM_USE_PETSC=ON "
        "(found MPI=${MFEM_USE_MPI}, PETSc=${MFEM_USE_PETSC})"
    )
endif()

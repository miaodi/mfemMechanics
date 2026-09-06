if(MFEM_USE_SUITESPARSE)
    mfem_mechanics_add_executable(
        PhaseField_shear
        shear.cpp
        mfemMechanics::femplugin
    )
else()
    message(STATUS "Skipping PhaseField_shear: installed MFEM reports MFEM_USE_SUITESPARSE=OFF")
endif()

if(MFEM_USE_MPI)
    mfem_mechanics_add_executable(
        pPhaseField_shear
        pshear.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
    if(NOT MFEM_USE_MUMPS)
        message(STATUS "pPhaseField_shear: direct solver unavailable (MFEM_USE_MUMPS=OFF); select -ls gmres")
    endif()
else()
    message(STATUS "Skipping pPhaseField_shear: installed MFEM reports MFEM_USE_MPI=OFF")
endif()

if(BUILD_TESTING AND TARGET PhaseField_shear)
    add_test(
        NAME PhaseFieldShear.SerialSmoke
        COMMAND PhaseField_shear -r 0 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1 -no-vis
    )
    set_tests_properties(
        PhaseFieldShear.SerialSmoke
        PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:PhaseField_shear>"
    )
endif()

if(BUILD_TESTING AND TARGET pPhaseField_shear)
    add_test(
        NAME PhaseFieldShear.OneRankSmoke
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 1
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:pPhaseField_shear>
            -ls gmres
            -rs 0 -rp 0 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1 -no-vis
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
        PhaseFieldShear.OneRankSmoke
        PROPERTIES
            PROCESSORS 1
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:pPhaseField_shear>"
    )
    add_test(
        NAME PhaseFieldShear.TwoRankSmoke
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:pPhaseField_shear>
            -ls gmres
            -rs 0 -rp 0 -disp 1e-9 -tf 1 -dt 1 -dt-max 1 -steps 1 -no-vis
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(
        PhaseFieldShear.TwoRankSmoke
        PROPERTIES
            PROCESSORS 2
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:pPhaseField_shear>"
    )
endif()

if(BUILD_TESTING AND TARGET pPhaseField_shear)
    foreach(ranks IN ITEMS 1 2)
        foreach(solver IN ITEMS direct gmres)
            if(solver STREQUAL "direct" AND NOT MFEM_USE_MUMPS)
                continue()
            endif()
            # Resolve the phase length scale sufficiently to retain admissible
            # nodal phase values at this load, and exercise tangent rebinding.
            add_test(
                NAME PhaseFieldShear.${solver}${ranks}RankRepeatedSolve
                COMMAND
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:pPhaseField_shear>
                    -m ${PROJECT_SOURCE_DIR}/data/crack_square2d_quad.msh
                    -ls ${solver} -il 1 -rs 4 -rp 0 -disp 1e-6
                    -tf 1 -dt 0.5 -dt-max 0.5 -steps 2 -no-vis
                    ${MPIEXEC_POSTFLAGS}
            )
            set_tests_properties(
                PhaseFieldShear.${solver}${ranks}RankRepeatedSolve
                PROPERTIES PROCESSORS ${ranks} TIMEOUT 120
                    WORKING_DIRECTORY "$<TARGET_FILE_DIR:pPhaseField_shear>"
            )
        endforeach()
    endforeach()
endif()

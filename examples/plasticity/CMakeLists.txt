mfem_mechanics_add_executable(j2_tensile j2_tensile.cpp mfemMechanics::femplugin)

if(MFEM_USE_MPI AND MFEM_USE_MUMPS)
    mfem_mechanics_add_executable(
        pCuProtrusion
        pCuProtrusion.cpp
        mfemMechanics::femplugin
        MPI::MPI_CXX
    )
else()
    message(STATUS
        "Skipping pCuProtrusion: requires MFEM_USE_MPI=ON and MFEM_USE_MUMPS=ON "
        "(found MPI=${MFEM_USE_MPI}, MUMPS=${MFEM_USE_MUMPS})"
    )
endif()

if(BUILD_TESTING)
    add_test(
        NAME J2Tensile.Smoke
        COMMAND j2_tensile -steps 2 -disp 0.01 -no-output
    )
    add_test(
        NAME J2Tensile.RefinementSmoke
        COMMAND j2_tensile -r 1 -lr 1 -steps 2 -disp 0.01 -no-output
    )
    set_tests_properties(
        J2Tensile.Smoke
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Plastic response: active.*unloading response: elastic"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:j2_tensile>"
    )
    set_tests_properties(
        J2Tensile.RefinementSmoke
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Plastic response: active"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:j2_tensile>"
    )

    if(TARGET pCuProtrusion)
        add_test(
            NAME CuProtrusion.OneRankSmoke
            COMMAND
                ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 1
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:pCuProtrusion>
                -steps 2
                -no-output
                ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(
            CuProtrusion.OneRankSmoke
            PROPERTIES
                PASS_REGULAR_EXPRESSION "Thermal J2 response: active.*protrusion response: upward"
                PROCESSORS 1
                WORKING_DIRECTORY "$<TARGET_FILE_DIR:pCuProtrusion>"
        )
        add_test(
            NAME CuProtrusion.TwoRankSmoke
            COMMAND
                ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:pCuProtrusion>
                -steps 2
                -no-output
                ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(
            CuProtrusion.TwoRankSmoke
            PROPERTIES
                PASS_REGULAR_EXPRESSION "Thermal J2 response: active.*protrusion response: upward"
                PROCESSORS 2
                WORKING_DIRECTORY "$<TARGET_FILE_DIR:pCuProtrusion>"
        )
    endif()
endif()

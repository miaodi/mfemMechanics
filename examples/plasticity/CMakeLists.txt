mfem_mechanics_add_executable(j2_tensile j2_tensile.cpp mfemMechanics::femplugin)
mfem_mechanics_add_executable(j2_bauschinger j2_bauschinger.cpp mfemMechanics::matlib)

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
    add_test(
        NAME J2Bauschinger.Smoke
        COMMAND j2_bauschinger -steps 1 -no-output
    )
    add_test(
        NAME J2Bauschinger.Csv
        COMMAND
            ${CMAKE_COMMAND}
            "-DEXECUTABLE=$<TARGET_FILE:j2_bauschinger>"
            "-DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/j2_bauschinger_test.csv"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_bauschinger_csv.cmake"
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
    set_tests_properties(
        J2Bauschinger.Smoke
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Bauschinger effect: active"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:j2_bauschinger>"
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

mfem_mechanics_add_executable(j2_tensile j2_tensile.cpp mfemMechanics::femplugin)

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
endif()

mfem_mechanics_add_executable(penalty_contact circle.cpp mfemMechanics::femplugin)

if(BUILD_TESTING)
    foreach(method IN ITEMS Penalty Semismooth)
        add_test(
            NAME ${method}Contact.LoadReturn
            COMMAND ${CMAKE_COMMAND}
                "-DEXECUTABLE=$<TARGET_FILE:penalty_contact>"
                "-DMETHOD=${method}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_load_return.cmake"
        )
        add_test(
            NAME ${method}Contact.OutputVerification
            COMMAND
                ${CMAKE_COMMAND}
                "-DEXECUTABLE=$<TARGET_FILE:penalty_contact>"
                "-DMETHOD=${method}"
                "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/${method}ContactOutput"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_contact_output.cmake"
        )
    endforeach()
    add_test(
        NAME PenaltyContact.Benchmark
        COMMAND penalty_contact -nx 2 -ny 2 -steps 2 -no-output
    )
    add_test(
        NAME PenaltyContact.RefinementBenchmark
        COMMAND penalty_contact -nx 2 -ny 2 -r 1 -steps 2 -no-output
    )
    add_test(
        NAME SemismoothContact.Benchmark
        COMMAND penalty_contact --semismooth -nx 2 -ny 2 -steps 4 -no-output
    )
    add_test(
        NAME SemismoothContact.CoarseQuadraticBenchmark
        COMMAND penalty_contact --semismooth -nx 2 -ny 2 -o 2 -steps 4 -no-output
    )
    add_test(
        NAME SemismoothContact.CoarseQuadraticHigherQuadrature
        COMMAND penalty_contact --semismooth -nx 2 -ny 2 -o 2 -steps 4 -cq 255 -no-output
    )
    add_test(
        NAME SemismoothContact.RefinementBenchmark
        COMMAND
            ${CMAKE_COMMAND}
            "-DEXECUTABLE=$<TARGET_FILE:penalty_contact>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_semismooth_refinement.cmake"
    )
    add_test(
        NAME SemismoothContact.UnderresolvedQuadratureRejected
        COMMAND
            ${CMAKE_COMMAND}
            "-DEXECUTABLE=$<TARGET_FILE:penalty_contact>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_underresolved_contact.cmake"
    )
    set_tests_properties(
        PenaltyContact.Benchmark
        PenaltyContact.RefinementBenchmark
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Penalty contact verification: passed"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:penalty_contact>"
    )
    set_tests_properties(
        SemismoothContact.Benchmark
        SemismoothContact.CoarseQuadraticBenchmark
        SemismoothContact.CoarseQuadraticHigherQuadrature
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Semismooth contact verification: passed"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:penalty_contact>"
    )
    set_tests_properties(
        SemismoothContact.RefinementBenchmark
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Semismooth refinement verification: passed"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:penalty_contact>"
    )
endif()

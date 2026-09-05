mfem_mechanics_add_executable(penalty_contact circle.cpp mfemMechanics::femplugin)

if(BUILD_TESTING)
    add_test(
        NAME PenaltyContact.Benchmark
        COMMAND penalty_contact -nx 2 -ny 2 -steps 2 -no-output
    )
    add_test(
        NAME PenaltyContact.RefinementBenchmark
        COMMAND penalty_contact -nx 2 -ny 2 -r 1 -steps 2 -no-output
    )
    set_tests_properties(
        PenaltyContact.Benchmark
        PenaltyContact.RefinementBenchmark
        PROPERTIES
            PASS_REGULAR_EXPRESSION "Penalty contact verification: passed"
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:penalty_contact>"
    )
endif()

if(NOT DEFINED EXECUTABLE)
    message(FATAL_ERROR "The underresolution regression requires EXECUTABLE.")
endif()

execute_process(
    COMMAND "${EXECUTABLE}" --semismooth -nx 2 -ny 2 -o 2 -steps 4 -cq 4 -no-output
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)
if(NOT result EQUAL 2 OR NOT output MATCHES "Semismooth contact verification: failed"
   OR NOT output MATCHES "Active contact quadrature points: 0/6"
   OR NOT error MATCHES "Independent gap samples detect penetration without active contact")
    message(FATAL_ERROR "Expected an explicitly rejected underresolved solve:\n${output}\n${error}")
endif()
string(REGEX MATCH "Independent sampled maximum penetration: ([+0-9.eE-]+)" match "${output}")
if(NOT match OR CMAKE_MATCH_1 LESS 0.00999 OR CMAKE_MATCH_1 GREATER 0.01001)
    message(FATAL_ERROR "Independent samples did not detect the rigid-motion penetration of 0.01:\n${output}")
endif()
message(STATUS "Underresolved contact correctly rejected; independent samples detect penetration.")

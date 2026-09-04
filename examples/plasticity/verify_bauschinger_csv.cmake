if(NOT DEFINED EXECUTABLE OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "The Bauschinger CSV test requires EXECUTABLE and OUTPUT_FILE.")
endif()

file(REMOVE "${OUTPUT_FILE}")
execute_process(
    COMMAND "${EXECUTABLE}" -steps 20 -f "${OUTPUT_FILE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE standard_output
    ERROR_VARIABLE standard_error
)
if(NOT result EQUAL 0)
    file(REMOVE "${OUTPUT_FILE}")
    message(FATAL_ERROR "j2_bauschinger failed:\n${standard_output}\n${standard_error}")
endif()
if(NOT EXISTS "${OUTPUT_FILE}")
    message(FATAL_ERROR "j2_bauschinger did not create its requested CSV file.")
endif()

file(STRINGS "${OUTPUT_FILE}" rows)
file(REMOVE "${OUTPUT_FILE}")
list(LENGTH rows row_count)
if(NOT row_count EQUAL 62)
    message(FATAL_ERROR "Expected 62 CSV lines, found ${row_count}.")
endif()

list(GET rows 0 header)
string(CONCAT expected_header
    "step,stage,axial_strain,isotropic_axial_stress,kinematic_axial_stress,"
    "isotropic_equivalent_plastic_strain,kinematic_equivalent_plastic_strain,"
    "kinematic_backstress_xx,isotropic_branch,kinematic_branch"
)
if(NOT "${header}" STREQUAL "${expected_header}")
    message(FATAL_ERROR "Unexpected Bauschinger CSV header: ${header}")
endif()
if(NOT standard_output MATCHES "Bauschinger effect: active")
    message(FATAL_ERROR "j2_bauschinger did not report the expected effect:\n${standard_output}")
endif()

message(STATUS "${standard_output}")

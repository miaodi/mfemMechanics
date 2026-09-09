if(NOT DEFINED EXECUTABLE)
    message(FATAL_ERROR "The semismooth refinement test requires EXECUTABLE.")
endif()

function(run_contact refinement output_variable)
    execute_process(
        COMMAND "${EXECUTABLE}" --semismooth -nx 2 -ny 2 -r "${refinement}" -steps 4 -no-output
        RESULT_VARIABLE result
        OUTPUT_VARIABLE standard_output
        ERROR_VARIABLE standard_error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "Semismooth contact failed at refinement ${refinement}:\n${standard_output}\n${standard_error}"
        )
    endif()
    if(NOT standard_output MATCHES "Semismooth contact verification: passed")
        message(FATAL_ERROR "Semismooth contact did not report success:\n${standard_output}")
    endif()
    set(${output_variable} "${standard_output}" PARENT_SCOPE)
endfunction()

function(extract_metric output label result_variable)
    string(REGEX MATCH "${label}: ([+0-9.eE-]+)" match "${output}")
    if(NOT match)
        message(FATAL_ERROR "Could not extract '${label}' from benchmark output:\n${output}")
    endif()
    set(${result_variable} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

run_contact(0 coarse_output)
run_contact(1 refined_output)
extract_metric("${coarse_output}" "Maximum penetration" coarse_penetration)
extract_metric("${refined_output}" "Maximum penetration" refined_penetration)
extract_metric("${coarse_output}" "Maximum pointwise contact-map residual" coarse_contact_map_residual)
extract_metric("${refined_output}" "Maximum pointwise contact-map residual" refined_contact_map_residual)
extract_metric("${coarse_output}" "Boundary P0 multiplier dofs" coarse_multiplier_count)
extract_metric("${refined_output}" "Boundary P0 multiplier dofs" refined_multiplier_count)

if(NOT refined_penetration LESS coarse_penetration)
    message(FATAL_ERROR
        "Uniform refinement did not reduce penetration: ${coarse_penetration} -> ${refined_penetration}."
    )
endif()
if(NOT refined_contact_map_residual LESS coarse_contact_map_residual)
    message(FATAL_ERROR
        "Uniform refinement did not reduce the pointwise contact-map residual: "
        "${coarse_contact_map_residual} -> ${refined_contact_map_residual}."
    )
endif()
if(NOT coarse_multiplier_count EQUAL 2 OR NOT refined_multiplier_count EQUAL 4)
    message(FATAL_ERROR
        "Unexpected boundary P0 multiplier counts: ${coarse_multiplier_count} -> ${refined_multiplier_count}."
    )
endif()

message(STATUS
    "Semismooth refinement verification: passed "
    "(penetration ${coarse_penetration} -> ${refined_penetration}, "
    "contact-map residual ${coarse_contact_map_residual} -> ${refined_contact_map_residual})."
)

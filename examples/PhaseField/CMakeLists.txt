project(PhaseField)
add_executable("${PROJECT_NAME}_shear" shear.cpp)
target_include_directories("${PROJECT_NAME}_shear" PUBLIC ${INCLUDE_DIRS})
target_link_libraries("${PROJECT_NAME}_shear" PRIVATE ${LIBS})
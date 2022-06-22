project(eigenbuckling)
add_executable(eigenbuckling main.cpp)
target_include_directories(eigenbuckling PUBLIC ${INCLUDE_DIRS})
target_link_libraries(eigenbuckling PRIVATE ${LIBS})
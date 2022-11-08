project(linear_tests)
add_executable(compositePlate compositePlate.cpp)
target_include_directories(compositePlate PUBLIC ${INCLUDE_DIRS})
target_link_libraries(compositePlate PRIVATE ${LIBS})

add_executable(compositePlateFull compositePlateFull.cpp)
target_include_directories(compositePlateFull PUBLIC ${INCLUDE_DIRS})
target_link_libraries(compositePlateFull PRIVATE ${LIBS})
cmake_minimum_required(VERSION 3.5...3.20)
project(tests)

include(FetchContent)

# Add Google Test
include(../cmake/GoogleTest.cmake)

add_executable(
    autodiff_test
    autodiff_test.cpp
)


target_include_directories(autodiff_test PUBLIC ${INCLUDE_DIRS})

target_link_libraries(
    autodiff_test PUBLIC GTest::gtest_main ${LIBS}
)

include(GoogleTest)
gtest_discover_tests(autodiff_test)
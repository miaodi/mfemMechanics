include("${PROJECT_SOURCE_DIR}/cmake/GoogleTest.cmake")
include("${CMAKE_ROOT}/Modules/GoogleTest.cmake")

mfem_mechanics_add_executable(
    autodiff_test
    autodiff_test.cpp
    GTest::gtest_main
    mfemMechanics::util
)
mfem_mechanics_add_executable(
    eigen_decomp_test
    eigen_decomp_test.cpp
    GTest::gtest_main
    mfemMechanics::util
)

gtest_discover_tests(autodiff_test)
gtest_discover_tests(eigen_decomp_test)

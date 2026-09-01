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
mfem_mechanics_add_executable(
    czm_history_test
    czm_history_test.cpp
    GTest::gtest_main
    mfemMechanics::femplugin
)

gtest_discover_tests(autodiff_test)
gtest_discover_tests(eigen_decomp_test)
gtest_discover_tests(czm_history_test)

add_library(util PrettyPrint.cpp util.cpp)
add_library(mfemMechanics::util ALIAS util)

target_compile_features(util PUBLIC cxx_std_17)
target_include_directories(util PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
    util
    PUBLIC
        Eigen3::Eigen
        MPI::MPI_CXX
        autodiff::autodiff
        mfem
)

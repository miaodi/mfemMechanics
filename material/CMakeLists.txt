add_library(matlib J2Plasticity.cpp Material.cpp NeoHookeanMaterial.cpp PhaseFieldMaterial.cpp)
add_library(mfemMechanics::matlib ALIAS matlib)

target_compile_features(matlib PUBLIC cxx_std_17)
target_include_directories(matlib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
    matlib
    PUBLIC
        mfemMechanics::util
        Eigen3::Eigen
        MPI::MPI_CXX
        autodiff::autodiff
        mfem
)

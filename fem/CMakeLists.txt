add_library(
    femplugin
    FEMPlugin.cpp
    PostProc.cpp
    StressFreeDeformation.cpp
    Solvers.cpp
    CZM.cpp
    AMR.cpp
    PhaseField.cpp
    J2PlasticityIntegrator.cpp
)
add_library(mfemMechanics::femplugin ALIAS femplugin)

target_compile_features(femplugin PUBLIC cxx_std_17)
target_include_directories(femplugin PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
    femplugin
    PUBLIC
        mfemMechanics::matlib
        mfemMechanics::util
        Eigen3::Eigen
        MPI::MPI_CXX
        autodiff::autodiff
        mfem
)

cmake_minimum_required(VERSION 3.20)

project(mfemMechanics LANGUAGES CXX)

include(FetchContent)
include(CTest)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(MFEM CONFIG REQUIRED)

if(NOT TARGET mfem)
    message(FATAL_ERROR "The installed MFEM package did not provide the required 'mfem' target")
endif()

find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen 3.4 package not found; fetching pinned Eigen 3.4.0")
    # Eigen is header-only; populate it without adding its tests, examples, or install rules.
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c
        SOURCE_SUBDIR _mfem_mechanics_fetch_only
    )
    FetchContent_MakeAvailable(eigen)

    add_library(mfemMechanics_eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS mfemMechanics_eigen)
    target_include_directories(mfemMechanics_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
endif()

find_package(autodiff CONFIG QUIET)
if(NOT TARGET autodiff::autodiff)
    message(STATUS "autodiff package not found; fetching pinned autodiff 1.1.2")
    # autodiff is header-only; avoid its upstream build, which also probes/enables CUDA.
    FetchContent_Declare(
        autodiff
        GIT_REPOSITORY https://github.com/autodiff/autodiff.git
        GIT_TAG 2e2f3a2b16afcd9c04e76c8a689e9fd23ff78679
        SOURCE_SUBDIR _mfem_mechanics_fetch_only
    )
    FetchContent_MakeAvailable(autodiff)

    add_library(mfemMechanics_autodiff INTERFACE)
    add_library(autodiff::autodiff ALIAS mfemMechanics_autodiff)
    target_compile_features(mfemMechanics_autodiff INTERFACE cxx_std_17)
    target_compile_definitions(mfemMechanics_autodiff INTERFACE AUTODIFF_EIGEN_FOUND)
    target_include_directories(mfemMechanics_autodiff SYSTEM INTERFACE "${autodiff_SOURCE_DIR}")
    target_link_libraries(mfemMechanics_autodiff INTERFACE Eigen3::Eigen)
endif()

find_package(OpenMP QUIET COMPONENTS CXX)

add_subdirectory(util)
add_subdirectory(material)
add_subdirectory(fem)

set(MFEM_MECHANICS_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${MFEM_MECHANICS_RUNTIME_OUTPUT_DIRECTORY}")
file(
    CREATE_LINK "${PROJECT_SOURCE_DIR}/data" "${MFEM_MECHANICS_RUNTIME_OUTPUT_DIRECTORY}/data"
    SYMBOLIC
    RESULT data_link_result
)
if(NOT data_link_result STREQUAL "0")
    message(FATAL_ERROR "Failed to create the runtime data symlink: ${data_link_result}")
endif()

function(mfem_mechanics_add_executable target source)
    add_executable("${target}" "${source}")
    target_compile_features("${target}" PRIVATE cxx_std_17)
    target_include_directories("${target}" PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries("${target}" PRIVATE ${ARGN})
    set_target_properties("${target}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MFEM_MECHANICS_RUNTIME_OUTPUT_DIRECTORY}")
endfunction()

mfem_mechanics_add_executable(test3 test3.cpp mfemMechanics::femplugin)
mfem_mechanics_add_executable(ex2 ex2.cpp mfemMechanics::femplugin)

if(MFEM_USE_SUITESPARSE)
    mfem_mechanics_add_executable(test2 test2.cpp mfemMechanics::femplugin)
    mfem_mechanics_add_executable(exec main.cpp mfemMechanics::femplugin)
    mfem_mechanics_add_executable(beam beam.cpp mfemMechanics::femplugin)
    mfem_mechanics_add_executable(block block.cpp mfemMechanics::femplugin)
else()
    message(STATUS "Skipping test2, exec, beam, and block: installed MFEM reports MFEM_USE_SUITESPARSE=OFF")
endif()

if(MFEM_USE_MPI)
    mfem_mechanics_add_executable(ex2p ex2p.cpp mfem MPI::MPI_CXX)
    mfem_mechanics_add_executable(pblock pblock.cpp mfemMechanics::femplugin MPI::MPI_CXX)
else()
    message(STATUS "Skipping ex2p: installed MFEM reports MFEM_USE_MPI=OFF")
    message(STATUS "Skipping pblock: installed MFEM reports MFEM_USE_MPI=OFF")
endif()

if(MFEM_USE_MPI AND MFEM_USE_PETSC)
    mfem_mechanics_add_executable(petchbuckle petchbuckle.cpp mfemMechanics::femplugin MPI::MPI_CXX)
else()
    message(STATUS
        "Skipping petchbuckle: requires MFEM_USE_MPI=ON and MFEM_USE_PETSC=ON "
        "(found MPI=${MFEM_USE_MPI}, PETSc=${MFEM_USE_PETSC})"
    )
endif()

add_subdirectory(examples)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

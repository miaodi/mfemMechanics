cmake_minimum_required(VERSION 3.16)
project(mfemMechanics)
include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_FIND_DEBUG_MODE 1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(MPI REQUIRED)
# FetchContent_Declare(
#     Fastor
#     GIT_REPOSITORY  https://github.com/romeric/Fastor.git
#     GIT_TAG         origin/master
# )

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/autodiff/autodiff.git
  GIT_TAG main
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

FetchContent_Declare(
  taco
  GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
  GIT_TAG master
)

set(BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(AUTODIFF_BUILD_TESTS OFF)
set(AUTODIFF_BUILD_PYTHON OFF)
set(AUTODIFF_BUILD_TESTS OFF)
set(AUTODIFF_BUILD_EXAMPLES OFF)

FetchContent_MakeAvailable(Eigen autodiff taco)

# FetchContent_Declare(
#   taco
#   GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
#   GIT_TAG master
# )

# FetchContent_MakeAvailable(taco)
# target_include_directories(taco PUBLIC ${taco_BINARY_DIR}/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MFEM_FOLDER "build-debug")
else()
    set(MFEM_FOLDER "build-release")
endif()

message("TACO_INCLUDE_DIR: ${TACO_INCLUDE_DIR}")
message("taco_BINARY_DIR: ${taco_BINARY_DIR}")

# find_path(TACO_INCLUDE_DIR REQUIRED NAMES taco.h HINTS ${CMAKE_BINARY_DIR}/_deps/taco-src/include)

# find_library(TACO_LIB REQUIRED NAMES taco libtaco HINTS ${CMAKE_BINARY_DIR}/lib/libtaco.so)

find_package(mfem REQUIRED NAMES MFEM  HINTS ${CMAKE_SOURCE_DIR}/../mfem/${MFEM_FOLDER})

add_subdirectory(util)
add_subdirectory(material)
add_subdirectory(fem)

set (INCLUDE_DIRS ${MFEM_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${UTIL_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR} ${TACO_INCLUDE_DIR})
set (LIBS ${MFEM_LIBRARIES} matlib femplugin MPI::MPI_CXX autodiff::autodiff Eigen3::Eigen ${CMAKE_BINARY_DIR}/lib/libtaco.so)

add_subdirectory(examples)

# add_executable(hello test.cpp)
# target_include_directories(hello PUBLIC ${INCLUDE_DIRS} ${fastor_SOURCE_DIR} ${TACO_INCLUDE_DIR})
# target_link_libraries(hello PRIVATE ${LIBS} ${TACO_LIB})

add_executable(test2 test2.cpp)
target_include_directories(test2 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(test2 PRIVATE ${LIBS})

add_executable(test3 test3.cpp)
target_include_directories(test3 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(test3 PRIVATE ${LIBS})

add_executable(exec main.cpp)
target_include_directories(exec PUBLIC ${INCLUDE_DIRS})
target_link_libraries(exec PRIVATE ${LIBS})


add_executable(petchbuckle petchbuckle.cpp)
target_include_directories(petchbuckle PUBLIC ${INCLUDE_DIRS})
target_link_libraries(petchbuckle PRIVATE ${LIBS})

# # add_executable(pexec pmain.cpp)
# # target_include_directories(pexec PUBLIC ${INCLUDE_DIRS})
# # target_link_libraries(pexec PRIVATE ${LIBS})

# add_executable(ex19 ex19.cpp)
# target_include_directories(ex19 PUBLIC ${INCLUDE_DIRS})
# target_link_libraries(ex19 PRIVATE ${LIBS})

# add_executable(ex16 ex16.cpp)
# target_include_directories(ex16 PUBLIC ${INCLUDE_DIRS})
# target_link_libraries(ex16 PRIVATE ${LIBS})

add_executable(ex2 ex2.cpp)
target_include_directories(ex2 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(ex2 PRIVATE ${LIBS})

# add_executable(ex2p ex2p.cpp)
# target_include_directories(ex2p PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
# target_link_libraries(ex2p PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(beam beam.cpp)
target_include_directories(beam PUBLIC ${INCLUDE_DIRS})
target_link_libraries(beam PRIVATE ${LIBS})

add_executable(block block.cpp)
target_include_directories(block PUBLIC ${INCLUDE_DIRS})
target_link_libraries(block PRIVATE ${LIBS})

add_executable(pblock pblock.cpp)
target_include_directories(pblock PUBLIC ${INCLUDE_DIRS})
target_link_libraries(pblock PRIVATE ${LIBS})


enable_testing()
add_subdirectory(tests)
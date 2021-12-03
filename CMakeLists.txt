cmake_minimum_required(VERSION 3.0)
project(test1)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

message(STATUS "Looking for mfem ...")
if(CMAKE_BUILD_TYPE EQUAL "DEBUG")
    set(MFEM_DIR "/home/dimiao/repo/mfem/debug")
else()
    set(MFEM_DIR "/home/dimiao/repo/mfem/release")
endif()

if (MFEM_DIR)
    find_package(mfem REQUIRED NAMES MFEM HINTS "${MFEM_DIR}"
            "${MFEM_DIR}/lib/cmake/mfem" NO_DEFAULT_PATH)
else()
    find_package(mfem REQUIRED NAMES MFEM)
endif()
message(STATUS "Found mfem config in: ${mfem_DIR} (version ${MFEM_VERSION})")
# Use the same C++ compiler as MFEM. This is needed when MFEM was built using
# an MPI wrapper and we do not have explicitly the MPI compile and link flags.
if (NOT CMAKE_CXX_COMPILER AND MFEM_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${MFEM_CXX_COMPILER}")
endif()

set(EIGEN_DIR "" CACHE PATH "/home/dimiao/repo/eigen-3.4.0")
find_package (Eigen3 3.3 REQUIRED NO_MODULE HINTS "${EIGEN_DIR}" "${EIGEN_DIR}/build")

add_subdirectory(material)
add_subdirectory(fem)

add_executable(test test.cpp)
target_include_directories(test PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(exec main.cpp)
target_include_directories(exec PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(exec PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(ex19 ex19.cpp)
target_include_directories(ex19 PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(ex19 PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

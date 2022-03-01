cmake_minimum_required(VERSION 3.0)
project(test1)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_FIND_DEBUG_MODE 1)

find_package(MPI REQUIRED)

find_package(mfem REQUIRED NAMES MFEM  HINTS ${CMAKE_SOURCE_DIR}/../mfem/release)

find_package (Eigen3 3.4 REQUIRED NO_MODULE HINTS ${CMAKE_SOURCE_DIR}/../eigen/build)


add_subdirectory(util)
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

add_executable(ex16 ex16.cpp)
target_include_directories(ex16 PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(ex16 PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(ex2 ex2.cpp)
target_include_directories(ex2 PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(ex2 PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

# add_executable(ex2p ex2p.cpp)
# target_include_directories(ex2p PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
# target_link_libraries(ex2p PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(beam beam.cpp)
target_include_directories(beam PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(beam PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(block block.cpp)
target_include_directories(block PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(block PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

# add_executable(pbeam pbeam.cpp)
# target_include_directories(pbeam PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
# target_link_libraries(pbeam PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

cmake_minimum_required(VERSION 3.0)
project(test1)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_FIND_DEBUG_MODE 1)

find_package(MPI REQUIRED)


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MFEM_FOLDER "debug")
else()
    set(MFEM_FOLDER "release")
endif()

find_package(mfem REQUIRED NAMES MFEM  HINTS ${CMAKE_SOURCE_DIR}/../mfem/${MFEM_FOLDER})

find_package (Eigen3 3.4 REQUIRED NO_MODULE HINTS ${CMAKE_SOURCE_DIR}/../eigen/build)

find_path(TACO_INCLUDE_DIRS REQUIRED NAMES taco.h HINTS ${CMAKE_SOURCE_DIR}/../taco/include)

find_library(TACO_LIB REQUIRED NAMES taco libtaco HINTS ${CMAKE_SOURCE_DIR}/../taco/build/lib)

add_subdirectory(util)
add_subdirectory(material)
add_subdirectory(fem)



set (INCLUDE_DIRS ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS} ${TACO_INCLUDE_DIRS})
set (LIBS ${MFEM_LIBRARIES} matlib femplugin ${TACO_LIB})

add_executable(test test.cpp)
target_include_directories(test PUBLIC ${INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${LIBS})

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

# add_executable(pexec pmain.cpp)
# target_include_directories(pexec PUBLIC ${INCLUDE_DIRS})
# target_link_libraries(pexec PRIVATE ${LIBS})

add_executable(ex19 ex19.cpp)
target_include_directories(ex19 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(ex19 PRIVATE ${LIBS})

add_executable(ex16 ex16.cpp)
target_include_directories(ex16 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(ex16 PRIVATE ${LIBS})

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
target_link_libraries(pblock PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

add_executable(pbeam pbeam.cpp)
target_include_directories(pbeam PUBLIC ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${PLUGIN_INCLUDE_DIRS})
target_link_libraries(pbeam PRIVATE ${MFEM_LIBRARIES} matlib femplugin)

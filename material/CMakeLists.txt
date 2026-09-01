project(material)

set(matSrc Material.cpp NeoHookeanMaterial.cpp PhaseFieldMaterial.cpp)

set(MATLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

add_library(matlib ${matSrc})

target_include_directories(matlib PUBLIC ${EIGEN3_INCLUDE_DIR} ${MFEM_INCLUDE_DIRS} ${UTIL_INCLUDE_DIRS})
target_link_libraries(matlib Eigen3::Eigen ${MFEM_LIBRARIES} util  MPI::MPI_CXX autodiff::autodiff)
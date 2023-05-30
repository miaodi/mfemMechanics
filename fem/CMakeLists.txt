project(plugin)

set(pluginSrc FEMPlugin.cpp PostProc.cpp Solvers.cpp CZM.cpp AMR.cpp)


set(PLUGIN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

add_library(femplugin ${pluginSrc})

target_include_directories(femplugin PUBLIC ${EIGEN3_INCLUDE_DIR} ${MFEM_INCLUDE_DIRS} ${MATLIB_INCLUDE_DIRS} ${UTIL_INCLUDE_DIRS} ${MPI_INCLUDE_DIRS})
target_link_libraries(femplugin Eigen3::Eigen ${MFEM_LIBRARIES} matlib util ${MPI_LIB} autodiff::autodiff)
project(util)

set(utilSrc PrettyPrint.cpp util.cpp)


set(UTIL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
add_library(util ${utilSrc})

target_include_directories(util PRIVATE ${EIGEN3_INCLUDE_DIR} ${MFEM_INCLUDE_DIRS} ${UTIL_INCLUDE_DIRS})
target_link_libraries(util Eigen3::Eigen ${MFEM_LIBRARIES} autodiff::autodiff)
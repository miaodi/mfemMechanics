project(util)

add_library(util INTERFACE)

target_include_directories(util INTERFACE ${EIGEN3_INCLUDE_DIR} ${MFEM_INCLUDE_DIRS} .)
# target_link_libraries(util Eigen3::Eigen ${MFEM_LIBRARIES} .)
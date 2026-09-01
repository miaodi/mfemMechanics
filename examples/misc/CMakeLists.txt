project(mis)
add_executable(autodiffPrac autodiffPrac.cpp)
target_include_directories(autodiffPrac PUBLIC ${INCLUDE_DIRS})
target_link_libraries(autodiffPrac PRIVATE ${LIBS})
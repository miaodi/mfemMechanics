project(blockOperator)
add_executable(ex19 ex19.cpp)
target_include_directories(ex19 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(ex19 PRIVATE ${LIBS})
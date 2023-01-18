project(czm)
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}2 main2.cpp)
target_include_directories(${PROJECT_NAME}2 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}2 PRIVATE ${LIBS})
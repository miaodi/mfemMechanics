project(postBuckling3D)
add_executable(${PROJECT_NAME}1 main1.cpp)
target_include_directories(${PROJECT_NAME}1 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}1 PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}2 main2.cpp)
target_include_directories(${PROJECT_NAME}2 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}2 PRIVATE ${LIBS})
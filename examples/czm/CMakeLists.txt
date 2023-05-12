project(czm)
add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}2 main2.cpp)
target_include_directories(${PROJECT_NAME}2 PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}2 PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}3D main3D.cpp)
target_include_directories(${PROJECT_NAME}3D PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}3D PRIVATE ${LIBS})

add_executable(bar bar.cpp)
target_include_directories(bar PUBLIC ${INCLUDE_DIRS})
target_link_libraries(bar PRIVATE ${LIBS})

add_executable(bar_3D bar_3D.cpp)
target_include_directories(bar_3D PUBLIC ${INCLUDE_DIRS})
target_link_libraries(bar_3D PRIVATE ${LIBS})

add_executable(test_3D_BUG test_3D_BUG.cpp)
target_include_directories(test_3D_BUG PUBLIC ${INCLUDE_DIRS})
target_link_libraries(test_3D_BUG PRIVATE ${LIBS})
project(heat)
add_executable(${PROJECT_NAME}_dynamic dynamic.cpp)
target_include_directories(${PROJECT_NAME}_dynamic PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_dynamic PRIVATE ${LIBS})

add_executable(${PROJECT_NAME}_static static.cpp)
target_include_directories(${PROJECT_NAME}_static PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_static PRIVATE ${LIBS})



add_executable(thermalStrain thermalStrain.cpp)
target_include_directories(thermalStrain PUBLIC ${INCLUDE_DIRS})
target_link_libraries(thermalStrain PRIVATE ${LIBS})


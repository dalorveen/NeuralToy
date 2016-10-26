cmake_minimum_required(VERSION 2.6)
project(NeuralToy)

add_definitions(-std=c++11)

set(LIBRARY_NAME neuralToy)
set(EXECUTABLE_NAME example)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/lib/include
)

file(
	GLOB
	INCLUDE_FILES
	lib/include/*
)

file(
	GLOB
	SRC_FILES
	lib/src/*
)

add_library(
    ${LIBRARY_NAME}
    STATIC
    ${SRC_FILES}
    ${INCLUDE_FILES}
)

add_executable(
	${EXECUTABLE_NAME}
	${CMAKE_CURRENT_SOURCE_DIR}/example/main.cpp
)

target_link_libraries(
    ${EXECUTABLE_NAME}
    ${LIBRARY_NAME}
)

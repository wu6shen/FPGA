cmake_minimum_required(VERSION 2.8)

project(test)

set(CMAKE_CXX_COMPLIER "g++")
set(CMAKE_CXX_FLAGS "-std=c++14 -march=native -O3")

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS 
	${OpenCV_LIBS} )

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/cereal ${PROJECT_SOURCE_DIR}/tiny_dnn ${PROJECT_SOURCE_DIR}/third_party ${PROJECT_SOURCE_DIR})

add_executable(main main.cc)

target_link_libraries( main ${OpenCV_LIBS} pthread)

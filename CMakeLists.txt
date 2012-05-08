
cmake_minimum_required(VERSION 2.8)
project(FreenectFusion)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#find_package(Boost 1.42.0 REQUIRED)
find_package(CUDA REQUIRED)
#find_package(OpenMP REQUIRED)

add_subdirectory(src)

cmake_minimum_required(VERSION 3.10.0)
project(matmul VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(matmul main.cpp
    matrix.h
    matrix.cpp)


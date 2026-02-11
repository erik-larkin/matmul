cmake_minimum_required(VERSION 3.10.0)
project(matmul)

find_package(benchmark REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

set(CMAKE_CXX_STANDARD 20)

add_executable(matmul main.cpp
    matrix.h
    matrix.cpp)

target_link_libraries(matmul PUBLIC
        benchmark::benchmark
        MKL::MKL
)

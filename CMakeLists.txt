cmake_minimum_required(VERSION 3.10.0)
project(matmul)

find_package(benchmark REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-mavx)

add_executable(matmul-bench
        benchmark.cpp
        matrix.h
        matrix.cpp
)

add_executable(matmul-main
        main.cpp
        matrix.cpp
)

target_link_libraries(matmul-bench PUBLIC
        benchmark::benchmark
        MKL::MKL
)

target_link_libraries(matmul-main PUBLIC
        benchmark::benchmark
        MKL::MKL
)

cmake_minimum_required(VERSION 3.10.0)
project(matmul)

find_package(benchmark REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-mavx)
#add_compile_options(-O3)

add_executable(matmul-bench
        benchmark.cpp
        matrix.h
        matrix.cpp
        outofplace.cpp
        outofplace.h
        inplace.cpp
        inplace.h
)

add_executable(matmul-main
        main.cpp
        matrix.cpp
        matrix.h
        outofplace.cpp
        outofplace.h
        inplace.cpp
        inplace.h
)

target_link_libraries(matmul-bench PUBLIC
        benchmark::benchmark
        MKL::MKL
)

target_link_libraries(matmul-main PUBLIC
        benchmark::benchmark
        MKL::MKL
)

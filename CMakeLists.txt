cmake_minimum_required(VERSION 3.10.0)
project(matmul)

find_package(benchmark REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/850cb69efda7ef00aed1c08fbe2bc1b6b4e615ee.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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

enable_testing()

add_executable(matmul-test
        test/outofplace_test.cpp
        matrix.cpp
        outofplace.cpp
)
target_link_libraries(matmul-test
        GTest::gtest_main
        MKL::MKL
)

include(GoogleTest)
gtest_discover_tests(matmul-test)
#include <iostream>
#include "matrix.h"
#include <benchmark/benchmark.h>
#include <mkl.h>

#define ORDER 32

static void BM_blas_gemm(benchmark::State& state) {
    Matrix A = gen_random_matrix(ORDER, ORDER);
    Matrix B = gen_random_matrix(ORDER, ORDER);

    for (auto _ : state) {
        blas_mul(A, B);
    }
}

BENCHMARK(BM_blas_gemm);

static void BM_naive_matmul(benchmark::State& state) {
    Matrix A = gen_random_matrix(ORDER, ORDER);
    Matrix B = gen_random_matrix(ORDER, ORDER);

    for (auto _ : state) {
        naive_mul(A, B);
    }
}

BENCHMARK(BM_naive_matmul);

static void BM_inplace_naive_matmul(benchmark::State& state) {
    Matrix A = gen_random_matrix(ORDER, ORDER);
    Matrix B = gen_random_matrix(ORDER, ORDER);

    for (auto _ : state) {
        in_place_mul(A, B);
    }
}

BENCHMARK(BM_inplace_naive_matmul);

BENCHMARK_MAIN();

// int main() {
//     Matrix A = gen_random_matrix(3, 3);
//     Matrix B = gen_random_matrix(3, 3);
//
//     Matrix C = naive_mul(A, B);
//     // in_place_mul(A, B);
//
//     Matrix C2 = blas_mul(A, B);
//
//     C.print();
//     C2.print();
//
//     const double diff = sum_abs_difference(C, C2);
//     std::cout << std::fixed;
//     std::cout << "Error: " << diff << std::endl;
//
//     return 0;
// }
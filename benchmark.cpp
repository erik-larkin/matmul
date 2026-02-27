#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"
#include <benchmark/benchmark.h>

#define ORDER 256

static void BM_blas_gemm(benchmark::State& state) {
    Matrix A = gen_random_matrix(ORDER, ORDER);
    Matrix B = gen_random_matrix(ORDER, ORDER);

    for (auto _ : state) {
        blas_mul(A, B);
    }
}

// static void BM_naive_matmul(benchmark::State& state) {
//     Matrix A = gen_random_matrix(ORDER, ORDER);
//     Matrix B = gen_random_matrix(ORDER, ORDER);
//
//     for (auto _ : state) {
//         naive_mul(A, B);
//     }
// }

// BENCHMARK(BM_naive_matmul);

// static void BM_inplace_naive_matmul(benchmark::State& state) {
//     Matrix A = gen_random_matrix(ORDER, ORDER);
//     Matrix B = gen_random_matrix(ORDER, ORDER);
//
//     for (auto _ : state) {
//         in_place_mul(A, B);
//     }
// }

// BENCHMARK(BM_inplace_naive_matmul);

static void BM_inplace_blas_matmul_rows(benchmark::State& state) {
    Matrix A = gen_random_matrix(ORDER, ORDER);
    Matrix B = gen_random_matrix(ORDER, ORDER);

    for (auto _ : state) {
        in_place_mul_rows(A, B, state.range(0));
    }
}

BENCHMARK(BM_inplace_blas_matmul_rows)->RangeMultiplier(2)->Range(1, ORDER);
BENCHMARK(BM_blas_gemm);

BENCHMARK_MAIN();

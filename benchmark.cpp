#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"
#include <benchmark/benchmark.h>

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

#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"
#include <benchmark/benchmark.h>
#include <mkl.h>

#define MAX_ORDER 4

static void BM_gemm_noalloc(benchmark::State& state) {
    const int order = state.range(0);
    state.counters.insert({{"Order", order}, {"PanelSize", order}});

    Matrix A = gen_random_matrix(order, order);
    Matrix B = gen_random_matrix(order, order);
    Matrix C(A.rows, B.cols);

    for (auto _ : state) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0,
            &A.data[0], A.cols, &B.data[0], B.cols, 0.0, &C.data[0], C.cols);
    }
}

static void BM_gemm_alloc(benchmark::State& state) {
    const int order = state.range(0);
    state.counters.insert({{"Order", order}, {"PanelSize", order}});

    Matrix A = gen_random_matrix(order, order);
    Matrix B = gen_random_matrix(order, order);

    for (auto _ : state) {
        Matrix C(A.rows, B.cols);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0,
            &A.data[0], A.cols, &B.data[0], B.cols, 0.0, &C.data[0], C.cols);
    }
}

static void BM_naive_mul(benchmark::State& state) {
    const int order = state.range(0);
    state.counters.insert({{"Order", order}, {"PanelSize", order}});

    Matrix A = gen_random_matrix(order, order);
    Matrix B = gen_random_matrix(order, order);

    for (auto _ : state) {
        naive_mul(A, B);
    }
}

static void BM_naive_in_place_mul(benchmark::State& state) {
    const int order = state.range(0);
    state.counters.insert({{"Order", order}, {"PanelSize", 1}});

    Matrix A = gen_random_matrix(order, order);
    Matrix B = gen_random_matrix(order, order);

    for (auto _ : state) {
        naive_in_place_mul(A, B);
    }
}

static void BM_inplace_gepm(benchmark::State& state) {
    const int order = state.range(0);
    const int panelSize = state.range(1);
    state.counters.insert({{"Order", order}, {"PanelSize", panelSize}});

    Matrix A = gen_random_matrix(order, order);
    Matrix B = gen_random_matrix(order, order);

    for (auto _ : state) {
        in_place_gepm(A, B, panelSize);
    }
}

static void BM_inplace_gemp(benchmark::State& state) {
    const int order = state.range(0);
    const int panelSize = state.range(1);
    state.counters.insert({{"Order", order}, {"PanelSize", panelSize}});

    Matrix A = gen_random_matrix(order, order);
    Matrix B = gen_random_matrix(order, order);

    for (auto _ : state) {
        in_place_gemp(A, B, panelSize);
    }
}

BENCHMARK(BM_naive_mul)->RangeMultiplier(2)->Range(1, 16);
BENCHMARK(BM_naive_in_place_mul)->RangeMultiplier(2)->Range(1, 16);

BENCHMARK(BM_gemm_noalloc)->RangeMultiplier(2)->Range(1, MAX_ORDER);
BENCHMARK(BM_gemm_alloc)->RangeMultiplier(2)->Range(1, MAX_ORDER);

static void OrderPanelSizePairs(benchmark::Benchmark* b) {
    for (int order = 1; order <= MAX_ORDER; order *= 2)
        for (int panelSize = 1; panelSize <= order; panelSize *= 2)
            b->Args({order, panelSize});
}

BENCHMARK(BM_inplace_gepm)->Apply(OrderPanelSizePairs);
BENCHMARK(BM_inplace_gemp)->Apply(OrderPanelSizePairs);

BENCHMARK_MAIN();

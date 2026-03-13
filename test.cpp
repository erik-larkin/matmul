#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"

using MatMulFunc = std::function<Matrix(Matrix&, Matrix&)>;

struct MatMulImpl {
    std::string name;
    MatMulFunc func;
};

bool matricesEqual(Matrix &A, Matrix &B, const double epsilon = 1e-6) {
    if (A.rows != B.rows) return false;
    if (A.cols != B.cols) return false;

    const double diff = sum_abs_difference(A, B);
    return diff < epsilon;
}

class MatrixMultiplicationTest : public ::testing::TestWithParam<MatMulImpl> {
protected:
    void SetUp() override {
        matmul = GetParam().func;
    }

    MatMulFunc matmul;
};

TEST_P(MatrixMultiplicationTest, IdentityMultiplication) {
    auto A = Matrix(std::vector<std::vector<double>>{{2, 3}, {4, 5}});
    auto id = Matrix(std::vector<std::vector<double>>{{1, 0}, {0, 1}});

    auto result = matmul(A, id);
    EXPECT_TRUE(matricesEqual(result, A));

    result = matmul(id, A);
    EXPECT_TRUE(matricesEqual(result, A));
}

TEST_P(MatrixMultiplicationTest, ZeroMultiplication) {
    auto A = Matrix(std::vector<std::vector<double>>{{1, 2}, {3, 4}});
    auto zeroes = Matrix(std::vector<std::vector<double>>{{0, 0}, {0, 0}});

    auto result = matmul(A, zeroes);
    EXPECT_TRUE(matricesEqual(result, zeroes));

    result = matmul(zeroes, A);
    EXPECT_TRUE(matricesEqual(result, zeroes));
}

TEST_P(MatrixMultiplicationTest, SquareMatrices) {
    auto A = Matrix(std::vector<std::vector<double>>{
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9,  10, 11, 12},
        {13, 14, 15, 16}
    });
    auto B = Matrix(std::vector<std::vector<double>>{
        {17, 18, 19, 20},
        {21, 22, 23, 24},
        {25, 26, 27, 28},
        {29, 30, 31, 32}
    });
    auto expected = Matrix(std::vector<std::vector<double>>{
        {250,  260,  270,  280},
        {618,  644,  670,  696},
        {986,  1028, 1070, 1112},
        {1354, 1412, 1470, 1528}
    });

    auto actual = matmul(A, B);
    EXPECT_TRUE(matricesEqual(actual, expected));
}

// this test assumes blas_mul is correct
TEST_P(MatrixMultiplicationTest, LargeMatrices) {
    auto A = gen_random_matrix(32, 32);
    auto B = gen_random_matrix(32, 32);
    auto expected = gemm(A, B);
    auto actual = matmul(A, B);
    EXPECT_TRUE(matricesEqual(actual, expected));
}

TEST_P(MatrixMultiplicationTest, NegativeNumbers) {
    auto A = Matrix(std::vector<std::vector<double>>{{-1, 2}, {3, -4}});
    auto B = Matrix(std::vector<std::vector<double>>{{5, -6}, {-7, 8}});
    auto expected = Matrix(std::vector<std::vector<double>>{
        {-19, 22},
        {43, -50}
    });

    auto actual = matmul(A, B);
    EXPECT_TRUE(matricesEqual(actual, expected));
}

TEST_P(MatrixMultiplicationTest, FloatingPoint) {
    auto A = Matrix(std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}});
    auto B = Matrix(std::vector<std::vector<double>>{{0.5, 0.6}, {0.7, 0.8}});
    auto expected = Matrix(std::vector<std::vector<double>>{
        {0.19, 0.22},
        {0.43, 0.50}
    });

    auto actual = matmul(A, B);
    EXPECT_TRUE(matricesEqual(actual, expected, 1e-9));
}

INSTANTIATE_TEST_SUITE_P(
    AllImplementations,
    MatrixMultiplicationTest,
    ::testing::Values(
        MatMulImpl{"Naive", naive_mul},
        MatMulImpl{"GEMM", gemm},
        MatMulImpl{"NaiveInPlace", naive_in_place_mul},
        MatMulImpl{"InPlaceGEPM", test_in_place_gepm},
        MatMulImpl{"InPlaceGEMP", test_in_place_gemp}
    ),
    [](const ::testing::TestParamInfo<MatMulImpl>& info) {
        return info.param.name;
    });
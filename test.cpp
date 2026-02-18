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

bool matricesEqual(Matrix &A, Matrix &B, const double epsilon = 1e-8) {
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
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    });
    auto B = Matrix(std::vector<std::vector<double>>{
        {10, 11, 12},
        {13, 14, 15},
        {16, 17, 18}
    });
    auto expected = Matrix(std::vector<std::vector<double>>{
        {84, 90, 96},
        {201, 216, 231},
        {318, 342, 366}
    });

    auto result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected));
}

TEST_P(MatrixMultiplicationTest, RectangularMatrices) {
    auto A = Matrix(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}});
    auto B = Matrix(std::vector<std::vector<double>>{{7, 8}, {9, 10}, {11, 12}});
    auto expected = Matrix(std::vector<std::vector<double>>{{58, 64}, {139, 154}});

    auto result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected));

    A = Matrix(std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}});  // 3x2
    B = Matrix(std::vector<std::vector<double>>{{7, 8, 9}, {10, 11, 12}});  // 2x3
    expected = Matrix(std::vector<std::vector<double>>{
        {27, 30, 33},
        {61, 68, 75},
        {95, 106, 117}
    });

    result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected));
}

TEST_P(MatrixMultiplicationTest, SingleElement) {
    auto A = Matrix(std::vector<std::vector<double>>{{5}});
    auto B = Matrix(std::vector<std::vector<double>>{{3}});

    auto expected = Matrix(std::vector<std::vector<double>>{{15}});
    auto result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected));
}

TEST_P(MatrixMultiplicationTest, RowVector_ColumnVector) {
    auto row = Matrix(std::vector<std::vector<double>>{{1, 2, 3}});
    auto col = Matrix(std::vector<std::vector<double>>{{4}, {5}, {6}});
    auto expected = Matrix(std::vector<std::vector<double>>{{32}});  // 1*4 + 2*5 + 3*6

    auto result = matmul(row, col);
    EXPECT_TRUE(matricesEqual(result, expected));
}

TEST_P(MatrixMultiplicationTest, ColumnVector_RowVector) {
    auto col = Matrix(std::vector<std::vector<double>>{{1}, {2}, {3}});
    auto row = Matrix(std::vector<std::vector<double>>{{4, 5, 6}});
    auto expected = Matrix(std::vector<std::vector<double>>{
        {4, 5, 6},
        {8, 10, 12},
        {12, 15, 18}
    });

    auto result = matmul(col, row);
    EXPECT_TRUE(matricesEqual(result, expected));
}

TEST_P(MatrixMultiplicationTest, NegativeNumbers) {
    auto A = Matrix(std::vector<std::vector<double>>{{-1, 2}, {3, -4}});
    auto B = Matrix(std::vector<std::vector<double>>{{5, -6}, {-7, 8}});
    auto expected = Matrix(std::vector<std::vector<double>>{
        {-19, 22},
        {43, -50}
    });

    auto result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected));
}

TEST_P(MatrixMultiplicationTest, FloatingPoint) {
    auto A = Matrix(std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}});
    auto B = Matrix(std::vector<std::vector<double>>{{0.5, 0.6}, {0.7, 0.8}});
    auto expected = Matrix(std::vector<std::vector<double>>{
        {0.19, 0.22},
        {0.43, 0.50}
    });

    auto result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected, 1e-9));
}

TEST_P(MatrixMultiplicationTest, LargerMatrix) {
    auto A = Matrix(std::vector<std::vector<double>>{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    });
    auto B = Matrix(std::vector<std::vector<double>>{
        {1, 0},
        {0, 1},
        {1, 0},
        {0, 1}
    });
    auto expected = Matrix(std::vector<std::vector<double>>{
        {4, 6},
        {12, 14},
        {20, 22}
    });
    auto result = matmul(A, B);
    EXPECT_TRUE(matricesEqual(result, expected));
}

// Dimension mismatch tests
TEST_P(MatrixMultiplicationTest, IncompatibleDimensions) {
    auto A = Matrix(std::vector<std::vector<double>>{{1, 2}, {3, 4}});  // 2x2
    auto B = Matrix(std::vector<std::vector<double>>{{1, 2, 3}});          // 1x3

    EXPECT_THROW(matmul(A, B), std::invalid_argument);
}

TEST_P(MatrixMultiplicationTest, EmptyMatrix) {
    auto A = Matrix(std::vector<std::vector<double>>{{1, 2}, {3, 4}});
    auto empty = Matrix(0, 0);

    EXPECT_THROW(matmul(A, empty), std::invalid_argument);
    EXPECT_THROW(matmul(empty, A), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    AllImplementations,
    MatrixMultiplicationTest,
    ::testing::Values(
        MatMulImpl{"Naive", naive_mul},
        MatMulImpl{"BLAS", blas_mul},
        MatMulImpl{"NaiveInPlace", in_place_mul}
    ),
    [](const ::testing::TestParamInfo<MatMulImpl>& info) {
        return info.param.name;
    });
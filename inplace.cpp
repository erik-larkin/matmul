#include "inplace.h"

#include <stdexcept>
#include <algorithm>
#include <mkl.h>

// in-place functions return the matrix that it overwrote in order to keep
// its interface the same as out-of-place variants, allowing both types
// of functions to use the test suite

Matrix in_place_mul(Matrix& A, Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    std::vector<double> A_row(A.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        const auto first = A.data.begin() + i * A.cols;
        const auto last = first + A.cols;
        std::ranges::copy(first, last, A_row.begin());
        std::ranges::fill(first, last, 0.0);

        for (size_t k = 0; k < A.cols; ++k) {
            const double A_ik = A_row[k];
            double total = 0.0;
            for (size_t j = 0; j < B.cols; ++j) {
                A.data[i * A.cols + j] += A_ik * B.e(k,j);
            }
        }
    }

    return A;
}

Matrix in_place_mul_rows(Matrix& A, Matrix& B, int rowsAtATime) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix A_rows(rowsAtATime, A.cols);

    const int leftoverRows = A.rows % rowsAtATime;
    size_t i;
    for (i = 0; i < A.rows - leftoverRows; i += rowsAtATime) {
        const auto first = A.data.begin() + i * A.cols;
        const auto last = first + A_rows.data.size();
        std::ranges::copy(first, last, A_rows.data.begin());
        std::ranges::fill(first, last, 0.0);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            A_rows.rows, B.cols, A_rows.cols, 1.0, &A_rows.data[0],
            A_rows.cols,&B.data[0], B.cols, 0.0, &A.data[i * A.cols], A.cols);
    }

    if (leftoverRows > 0) {
        const auto first = A.data.begin() + i * A.cols;
        const auto last = first + (leftoverRows * A.cols);
        std::ranges::copy(first, last, A_rows.data.begin());
        std::ranges::fill(first, last, 0.0);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            A_rows.rows, B.cols, A_rows.cols, 1.0, &A_rows.data[0],
            A_rows.cols,&B.data[0], B.cols, 0.0, &A.data[i * A.cols], A.cols);
    }

    return A;
}

Matrix test_in_place_mul_rows(Matrix& A, Matrix& B) {
    return in_place_mul_rows(A, B, 2);
}
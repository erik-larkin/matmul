#include "inplace.h"

#include <algorithm>
#include <cassert>
#include <mkl.h>

// in-place functions return the matrix that it overwrote in order to keep
// its interface the same as out-of-place variants, allowing both types
// of functions to use the test suite

Matrix& naive_in_place_mul(Matrix& A, Matrix& B) {
    Matrix C(1, A.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        std::ranges::fill(C.data.begin(), C.data.end(), 0.0);
        for (size_t k = 0; k < A.cols; ++k) {
            const auto A_ik = A.e(i, k);
            for (size_t j = 0; j < A.cols; ++j) {
                C.data[j] += A_ik * B.e(k,j);
            }
        }
        double* addr = &A.data[i*A.cols];
        std::ranges::copy(C.data.begin(), C.data.end(), addr);
    }

    return A;
}

inline Matrix& in_place_gexx(Matrix& A, Matrix& B, const int m, const int n, const bool overwriteB) {
    assert(A.rows % m == 0);
    assert(B.cols % n == 0);

    Matrix& R = overwriteB ? B : A;
    Matrix C(m, n);

    for (size_t i = 0; i < A.rows; i += C.rows) {
        for (size_t j = 0; j < B.cols; j += C.cols) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                C.rows, C.cols, A.cols, 1.0, &A.data[i * A.cols],
                A.cols,&B.data[j], B.cols, 0.0, &C.data[0], C.cols);

            mkl_domatcopy('R', 'N', C.rows, C.cols, 1.0,
                &C.data[0], C.cols, &R.data[(i * R.cols) + j], A.cols);
        }
    }

    return R;
}

Matrix& in_place_gepm(Matrix& A, Matrix& B, const int m) {
    return in_place_gexx(A, B, m, A.cols, false);
}

Matrix& test_in_place_gepm(Matrix& A, Matrix& B) {
    return in_place_gepm(A, B, 2);
}

Matrix& in_place_gemp(Matrix& A, Matrix& B, const int n) {
    return in_place_gexx(A, B, A.rows, n, true);
}

Matrix& test_in_place_gemp(Matrix& A, Matrix& B) {
    return in_place_gemp(A, B, 2);
}
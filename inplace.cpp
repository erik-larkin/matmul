#include "inplace.h"

#include <algorithm>
#include <cassert>
#include <mkl.h>

// in-place functions return the matrix that it overwrote in order to keep
// its interface the same as out-of-place variants, allowing both types
// of functions to use the test suite

Matrix& naive_in_place_mul(Matrix& A, Matrix& B) {
    Matrix C(1, B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        std::ranges::fill(C.data.begin(), C.data.end(), 0.0);
        for (size_t k = 0; k < A.cols; ++k) {
            const auto A_ik = A.e(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                C.data[j] += A_ik * B.e(k,j);
            }
        }
        double* addr = &A.data[i*A.cols];
        std::ranges::copy(C.data.begin(), C.data.end(), addr);
    }

    return A;
}

// assumes A.rows % m == 0

inline Matrix& in_place_gexx(Matrix& A, Matrix& B, Matrix& aux, const bool overwriteB) {
    assert(A.rows % aux.rows == 0);
    assert(B.cols % aux.cols == 0);

    Matrix& R = overwriteB ? B : A;

    for (size_t i = 0; i < A.rows; i += aux.rows) {
        for (size_t j = 0; j < B.cols; j += aux.cols) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                aux.rows, aux.cols, A.cols, 1.0, &A.data[i * A.cols],
                A.cols,&B.data[j], B.cols, 0.0, &aux.data[0], aux.cols);

            mkl_domatcopy('R', 'N', aux.rows, aux.cols, 1.0,
                &aux.data[0], aux.cols, &R.data[(i * R.cols) + j], A.cols);
        }
    }

    return R;
}

Matrix& in_place_gepm(Matrix& A, Matrix& B, Matrix& aux) {
    return in_place_gexx(A, B, aux, false);
}

Matrix& test_in_place_gepm(Matrix& A, Matrix& B) {
    Matrix aux(1, B.cols);
    return in_place_gepm(A, B, aux);
}

Matrix& in_place_gemp(Matrix& A, Matrix& B, Matrix& aux) {
    return in_place_gexx(A, B, aux, true);
}

Matrix& test_in_place_gemp(Matrix& A, Matrix& B) {
    Matrix aux(A.rows, 1);
    return in_place_gemp(A, B, aux);
}
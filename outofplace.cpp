#include "outofplace.h"

#include <mkl.h>

Matrix naive_mul(Matrix& A, Matrix& B) {
    Matrix R(A.rows, B.cols);

    for (size_t i = 0; i < R.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            const auto A_ik = A.e(i, k);
            for (size_t j = 0; j < R.cols; ++j) {
                R.data[i * R.cols + j] += A_ik * B.e(k, j);
            }
        }
    }

    return R;
}

Matrix gemm(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0,
        &A.data[0], A.cols, &B.data[0], B.cols, 0.0, &C.data[0], C.cols);

    return C;
}

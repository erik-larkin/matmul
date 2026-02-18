#include "outofplace.h"

#include <stdexcept>
#include <immintrin.h>
#include <mkl.h>

Matrix naive_mul(Matrix& A, Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix R(A.rows, B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            const double A_ik = A.e(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                R.data[i * R.cols + j] += A_ik * B.e(k, j);
            }
        }
    }

    return R;
}

Matrix vectorised_mul(Matrix& A, Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix R(A.rows, B.cols);

    __m256d r4;
    __m256d a4;
    __m256d b4;

    for (size_t i = 0; i < A.rows; ++i) {
        size_t k = 0;

        for (; k < A.cols - 3; k += 4) {
            a4 = _mm256_load_pd(&A.e(i, k));
            size_t j = 0;

            for (; j < B.cols - 3; j += 4) {
                r4 = _mm256_load_pd(&R.e(i, j));
                b4 = _mm256_load_pd(&B.e(k, j));
                b4 = _mm256_mul_pd(a4, b4);
                r4 = _mm256_add_pd(r4, b4);
                _mm256_store_pd(&R.e(i, j), r4);
            }

            const double A_ik = A.e(i, k);
            for (; j < B.cols; ++j) {
                R.data[i * R.cols + j] += A_ik * B.e(k, j);
            }
        }

        for (; k < A.cols; ++k) {
            const double A_ik = A.e(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                R.data[i * R.cols + j] += A_ik * B.e(k, j);
            }
        }
    }

    return R;
}

Matrix blas_mul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix C(A.rows, B.cols);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0,
        &A.data[0], A.cols, &B.data[0], B.cols, 0.0, &C.data[0], C.cols);

    return C;
}

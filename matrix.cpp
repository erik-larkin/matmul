#include "matrix.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>
#include <mkl.h>
#include <immintrin.h>

Matrix::Matrix(const size_t r, const size_t c) : data(r * c, 0.0), rows(r), cols(c) {}

Matrix::Matrix(const std::vector<std::vector<double>>& input) {
    if (input.empty() || input[0].empty()) {
        throw std::invalid_argument("Input matrix cannot be empty");
    }

    std::vector<double> result;
    for (const std::vector<double>& row : input) {
        result.insert(result.end(), row.begin(), row.end());
    }

    rows = input.size();
    cols = input[0].size();
    data = result;
}

Matrix::Matrix(const size_t r, const size_t c, const std::vector<double> &input): data(input), rows(r), cols(c) {}

double inline & Matrix::e(const size_t i, const size_t j) {
    return data[i * cols + j];
}

void inline Matrix::setE(const size_t i, const size_t j, const double value) {
    data[i * cols + j] = value;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

double sum_abs_difference(Matrix &A, Matrix &B) {
    Matrix R = A - B;

    double sum = 0;
    for (size_t i = 0; i < R.rows; ++i) {
        for (size_t j = 0; j < R.cols; ++j) {
            sum += fabs(R.e(i, j));
        }
    }

    return sum;
}

Matrix gen_random_matrix(int N, int M) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers

    std::uniform_real_distribution<double> dist {-512.0, 512.0};

    auto gen = [&](){
        return dist(mersenne_engine);
    };

    std::vector<double> vec(N * M);
    std::generate(vec.begin(), vec.end(), gen);

    Matrix result(N, M, vec);

    return result;
}

Matrix operator-(Matrix& A, Matrix& B) {
    if (A.cols != B.cols && A.rows != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix R(A.rows, A.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            const double val = A.e(i,j) - B.e(i,j);
            R.setE(i, j, val);
        }
    }

    return R;
}

Matrix blas_mul(const Matrix& A, const Matrix& B) {
    Matrix C(A.rows, B.cols);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, A.cols, 1.0,
        &A.data[0], A.cols, &B.data[0], B.cols, 0.0, &C.data[0], C.cols);

    return C;
}

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

void in_place_mul(Matrix& A, Matrix& B) {
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
}

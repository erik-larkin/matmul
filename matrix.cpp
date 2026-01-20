#include "matrix.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

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

Matrix naive_mul(Matrix& A, Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix R(A.rows, B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            const double A_ik = A.e(i, k);
            for (size_t j = 0; j < B.cols; ++j) {
                R.data[i * R.cols + j] += A_ik * B.e(k,j);
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

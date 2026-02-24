#include "matrix.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>

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

// assumes good alignment
Matrix Matrix::get_block(const size_t i, const size_t j, const size_t block_size) {
    std::vector<double> data;

    for (int r = 0; r < block_size; ++r) {
        double * addr = &e(i + r, j);
        data.insert(data.end(), addr, addr + block_size);
    }

    return Matrix(block_size, block_size, data);
}

void Matrix::set_block(const size_t i, const size_t j, Matrix& block) {
    for (int r = 0; r < block.rows; ++r) {
        for (int c = 0; c < block.cols; ++c) {
            setE(i + r, j + c, block.e(r, c));
        }
    }
}

void Matrix::operator+=(Matrix &other) {
    if (cols != other.cols && rows != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const double val = e(i,j) + other.e(i,j);
            setE(i, j, val);
        }
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

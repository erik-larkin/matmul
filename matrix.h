#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {

public:
    std::vector<double> data;
    size_t rows;
    size_t cols;

    Matrix(size_t r, size_t c);
    explicit Matrix(const std::vector<std::vector<double>>& input);
    explicit Matrix(size_t r, size_t c, const std::vector<double>& input);

    double& e(size_t i, size_t j);
    void setE(size_t i, size_t j, double value);
    void print() const;
    Matrix get_block(size_t i, size_t j, size_t block_size);
    void set_block(size_t i, size_t j, Matrix &block);

    void operator+=(Matrix& other);
};

double sum_abs_difference(Matrix& A, Matrix& B);
Matrix operator-(Matrix& A, Matrix& B);
Matrix gen_random_matrix(int N, int M);

#endif //MATRIX_H

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
    double& e(size_t i, size_t j);
    void setE(size_t i, size_t j, double value);
    void print() const;
};

Matrix naive_mul(Matrix& A, Matrix& B);
void in_place_mul(Matrix& A, Matrix& B);

#endif //MATRIX_H

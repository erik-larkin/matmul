#include <iostream>
#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"

int main() {
    Matrix A = gen_random_matrix(3, 3);
    Matrix B = gen_random_matrix(3, 3);

    Matrix C = blas_mul(A, B);
    Matrix C1 = naive_mul(A, B);
    // Matrix C2 = vectorised_mul(A, B);

    const double diff = sum_abs_difference(C, C1);
    std::cout << std::fixed;
    std::cout << "Error: " << diff << std::endl;

    return 0;
}
#include <iostream>
#include "matrix.h"

int main() {
    Matrix A = gen_random_matrix(32, 32);
    Matrix B = gen_random_matrix(32, 32);

    Matrix C = blas_mul(A, B);
    Matrix C1 = naive_mul(A, B);
    Matrix C2 = optimised_naive_mul(A, B);

    std::cout << "Blas" << std::endl;
    C.print();

    std::cout << std::endl << "Naive" << std::endl;
    C1.print();

    std::cout << std::endl << "Optimised naive" << std::endl;
    C2.print();

    const double diff = sum_abs_difference(C, C2);
    std::cout << std::fixed;
    std::cout << "Error: " << diff << std::endl;

    return 0;
}
#include <iostream>
#include "matrix.h"

int main() {
    Matrix A = gen_random_matrix(3, 3);
    Matrix B = gen_random_matrix(3, 3);

    Matrix C = naive_mul(A, B);
    in_place_mul(A, B);

    const double diff = sum_abs_difference(A, C);
    std::cout << "Error: " << diff << std::endl;

    return 0;
}
#include <iostream>
#include <regex>

#include "matrix.h"

int main() {
    Matrix A({{1, 2},
              {4, 5}});

    Matrix B({{7, 8},
              {9, 10}});

    Matrix C = naive_mul(A, B);

    std::cout << "Matrix A:" << std::endl;
    A.print();

    std::cout << "\nMatrix B:" << std::endl;
    B.print();

    std::cout << "\nMatrix C = A * B:" << std::endl;
    C.print();

    in_place_mul(A, B);
    std::cout << "\n(inplace) Matrix A = A * B:" << std::endl;
    A.print();

    return 0;
}
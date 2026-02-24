#include <iostream>
#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"

int main() {
    Matrix A = gen_random_matrix(2, 2);
    Matrix B = gen_random_matrix(2, 2);

    A.print();
    std::cout << std::endl;
    B.print();
    std::cout << std::endl;

    A += B;

    A.print();

    return 0;
}
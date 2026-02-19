#include <iostream>
#include "matrix.h"
#include "outofplace.h"
#include "inplace.h"

int main() {
    Matrix A = gen_random_matrix(4, 4);
    Matrix B = A.get_block(1, 1, 2);

    A.print();
    std::cout << std::endl;
    B.print();

    return 0;
}
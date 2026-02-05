#include <iostream>
#include <random>

#include "matrix.h"

Matrix gen_random_matrix(int N, int M) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers

    std::uniform_real_distribution<double> dist {1.0, 52.0};

    auto gen = [&](){
        return dist(mersenne_engine);
    };

    std::vector<double> vec(N * M);
    std::generate(vec.begin(), vec.end(), gen);

    Matrix result(N, M, vec);

    return result;
}

int main() {
    Matrix A = gen_random_matrix(3, 3);

    Matrix B = gen_random_matrix(3, 3);

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
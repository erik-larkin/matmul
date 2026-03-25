#ifndef INPLACE_H
#define INPLACE_H

#include "matrix.h"

Matrix& naive_in_place_mul(Matrix& A, Matrix& B);

// A cols * B rows => C all (partial)
// Matrix& in_place_gepp(Matrix& A, Matrix& B, int n);
// Matrix& test_in_place_gepp(Matrix& A, Matrix& B);

// A all * B cols => C cols
Matrix& in_place_gemp(Matrix& A, Matrix& B, Matrix& aux);
Matrix& test_in_place_gemp(Matrix& A, Matrix& B);

// A rows * B all => C rows
Matrix& in_place_gepm(Matrix& A, Matrix& B, Matrix& aux);
Matrix& test_in_place_gepm(Matrix& A, Matrix& B);

// // A block * B rows => C rows (partial)
// Matrix& in_place_gebp(Matrix& A, Matrix& B, int n);
// Matrix& test_in_place_gebp(Matrix& A,  Matrix& B);
//
// // A cols * B block => C cols (partial)
// Matrix& in_place_gepb(Matrix& A, Matrix& B, int n);
// Matrix& test_in_place_gepb(Matrix& A, Matrix& B);
//
// // A rows * B cols => C blocks
// Matrix& in_place_gepdot(Matrix& A, Matrix& B, int n);
// Matrix& test_in_place_gepdot(Matrix& A, Matrix& B);
//
// // A block * B block => C block (partial)
// Matrix& in_place_gebb(Matrix& A, Matrix& B, int n);
// Matrix& test_in_place_gebb(Matrix& A, Matrix& B);

#endif //INPLACE_H

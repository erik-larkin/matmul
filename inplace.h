#ifndef INPLACE_H
#define INPLACE_H

#include "matrix.h"

Matrix in_place_mul(Matrix& A, Matrix& B);
Matrix in_place_mul_rows(Matrix& A, Matrix& B, int rowsAtATime);
Matrix test_in_place_mul_rows(Matrix& A, Matrix& B);

#endif //INPLACE_H

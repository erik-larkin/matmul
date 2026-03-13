#ifndef OUTOFPLACE_H
#define OUTOFPLACE_H

#include "matrix.h"

Matrix naive_mul(Matrix& A, Matrix& B);
Matrix gemm(const Matrix& A, const Matrix& B);

#endif //OUTOFPLACE_H

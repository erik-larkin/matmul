#ifndef OUTOFPLACE_H
#define OUTOFPLACE_H

#include "matrix.h"

Matrix naive_mul(Matrix& A, Matrix& B);
Matrix blocked_mul(Matrix& A, Matrix& B);
Matrix vectorised_mul(Matrix& A, Matrix& B);
Matrix blas_mul(const Matrix& A, const Matrix& B);

#endif //OUTOFPLACE_H

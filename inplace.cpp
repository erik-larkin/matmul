#include "inplace.h"

#include <stdexcept>
#include <algorithm>

void in_place_mul(Matrix& A, Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    std::vector<double> A_row(A.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        const auto first = A.data.begin() + i * A.cols;
        const auto last = first + A.cols;
        std::ranges::copy(first, last, A_row.begin());
        std::ranges::fill(first, last, 0.0);

        for (size_t k = 0; k < A.cols; ++k) {
            const double A_ik = A_row[k];
            double total = 0.0;
            for (size_t j = 0; j < B.cols; ++j) {
                A.data[i * A.cols + j] += A_ik * B.e(k,j);
            }
        }
    }
}

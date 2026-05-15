#pragma once
#include <vector>

struct CSR_Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;

    int nnz() const { return values.size(); }
};
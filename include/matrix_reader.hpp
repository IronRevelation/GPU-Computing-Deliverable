#pragma once

#include "types.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

CSR_Matrix read_matrix(const std::string &filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        print_fatal_error("Error opening file " + filename);
    }

    CSR_Matrix matrix;

    std::string banner, object, format, field, symmetry;
    in >> banner >> object >> format >> field >> symmetry;

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> data;

    std::istringstream size_line(line);

    if (format == "coordinate") {
        int nnz;
        size_line >> matrix.rows >> matrix.cols >> nnz;

        for (int k = 0; k < nnz; ++k) {
            int i, j;
            float value;
            in >> i >> j >> value;
            row_indices.push_back(i - 1);
            col_indices.push_back(j - 1);
            data.push_back(value);

            if (symmetry == "symmetric" && i != j) {
                row_indices.push_back(j - 1);
                col_indices.push_back(i - 1);
                data.push_back(value);
            }
        }

    } else if (format == "array") {
        size_line >> matrix.rows >> matrix.cols;
        for (int i = 0; i < matrix.rows; ++i) {
            for (int j = 0; j < matrix.cols; ++j) {
                float value;
                in >> value;
                row_indices.push_back(i);
                col_indices.push_back(j);
                data.push_back(value);
            }
        }
    }

    matrix.row_offsets.assign(matrix.rows + 1, 0);
    matrix.col_indices.resize(data.size());
    matrix.values.resize(data.size());

    for (int row : row_indices) {
        matrix.row_offsets[row + 1] += 1;
    }

    for (int row = 0; row < matrix.rows; ++row) {
        matrix.row_offsets[row + 1] += matrix.row_offsets[row];
    }

    std::vector<int> next = matrix.row_offsets;

    for (int k = 0; k < data.size(); ++k) {
        int row = row_indices[k];
        int col = col_indices[k];
        int dest = next[row];
        next[row] += 1;
        matrix.col_indices[dest] = col;
        matrix.values[dest] = data[k];
    }

    return matrix;
}
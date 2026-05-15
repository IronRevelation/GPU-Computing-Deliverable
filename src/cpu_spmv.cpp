#include "cpu_spmv.hpp"

#include "types.hpp"
#include <chrono>
#include <vector>

void cpu_spmv_impl(const CSR_Matrix &csr, const std::vector<float> &x, std::vector<float> &result) {

    for (int row = 0; row < csr.rows; ++row) {
        result[row] = 0.0F;
        for (int col = csr.row_offsets[row]; col < csr.row_offsets[row + 1]; ++col) {
            result[row] += csr.values[col] * x[csr.col_indices[col]];
        }
    }
}

std::vector<float> cpu_spmv(const CSR_Matrix &csr, const std::vector<float> &x) {
    std::vector<float> y(csr.rows);

    cpu_spmv_impl(csr, x, y);
    return y;
}

std ::vector<double> cpu_spmv_timed(const CSR_Matrix &csr, const std::vector<float> &x, int warmup, int repeat) {
    std::vector<double> execution_times;
    std::vector<float> y(csr.rows);

    for (int i = 0; i < warmup + repeat; ++i) {

        auto start = std::chrono::steady_clock::now();
        cpu_spmv_impl(csr, x, y);

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        if (i >= warmup) {
            execution_times.push_back(elapsed.count());
        }
    }
    return execution_times;
}
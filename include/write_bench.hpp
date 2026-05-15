#pragma once

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

void write_benchmark_header(const std::string &path) {
    std::ofstream out(path, std::ios::trunc);
    out << "matrix,implementation,rows,cols,nnz,avg_nnz_per_row,long_rows,warmup_runs,measured_runs,mean_ms,stddev_ms,"
           "min_ms,gflops,effective_bandwidth_gb_s\n";
}

void write_benchmark(const std::string &path, std::string &matrix_name, const std::string &implementation_name,
                     CSR_Matrix &matrix, int warmup, int repeat, const std::vector<double> &timings) {
    std::ofstream out(path, std::ios::app);

    double avg_nnz_per_row = (double)(matrix.nnz()) / matrix.rows;

    constexpr int long_row_threshold = 256;

    int long_rows_count = 0;

    for (int row = 0; row < matrix.rows; ++row) {
        int row_nnz = matrix.row_offsets[row + 1] - matrix.row_offsets[row];

        if (row_nnz > long_row_threshold) {
            ++long_rows_count;
        }
    }

    double mean_ms = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();

    double variance_ms = 0.0;
    for (float t : timings) {
        double diff = t - mean_ms;
        variance_ms += diff * diff;
    }
    variance_ms /= (double)(timings.size() - 1);
    double stddev_ms = std::sqrt(variance_ms);

    double min_ms = *std::min_element(timings.begin(), timings.end());

    double seconds = mean_ms / 1000.0;

    // assuming two flop per nnz
    double gflops = (2.0 * (double)(matrix.values.size())) / seconds / 1e9;

    // 12 B/nnz for value, column index,and x read;
    // 12 B/row for y write and row_offsets reads.
    // Adaptive adds 4 B/row for the row-list read.
    double bytes = 12.0 * (double)(matrix.nnz()) + 12.0 * (double)(matrix.rows);
    if (implementation_name == "adaptive") {
        bytes += 4.0 * (double)(matrix.rows);
    }
    double bandwidth_gb_s = bytes / seconds / 1e9;

    out << std::setprecision(12) << matrix_name << "," << implementation_name << "," << matrix.rows << ","
        << matrix.cols << "," << matrix.nnz() << "," << avg_nnz_per_row << "," << long_rows_count << "," << warmup
        << "," << repeat << "," << mean_ms << "," << stddev_ms << "," << min_ms << "," << gflops << ","
        << bandwidth_gb_s << "\n";
}
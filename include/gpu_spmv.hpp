#pragma once

#include "types.hpp"

#include <vector>

std::vector<double> simple_gpu_spmv(const CSR_Matrix &csr, const std::vector<float> &x, std::vector<float> &y,
                                    int warmup = 0, int repeat = 1);
std::vector<double> adaptive_gpu_spmv(const CSR_Matrix &csr, const std::vector<float> &x, std::vector<float> &y,
                                      int warmup = 0, int repeat = 1);

std::vector<double> cusparse_gpu_spmv(const CSR_Matrix &csr, const std::vector<float> &x, std::vector<float> &y,
                                      int warmup = 0, int repeat = 1);

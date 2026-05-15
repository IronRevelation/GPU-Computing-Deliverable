#pragma once

#include "types.hpp"

#include <vector>

std::vector<float> cpu_spmv(const CSR_Matrix &csr, const std::vector<float> &x);
std ::vector<double> cpu_spmv_timed(const CSR_Matrix &csr, const std::vector<float> &x, int warmup = 0, int repeat = 1);

#include "cuda_utils.cuh"
#include <cuda_runtime.h>

#include "types.hpp"
#include <vector>

__global__ void simple_spmv_kernel(int rows, const int *row_offsets, const int *col_indicies, const float *values,
                                   const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0F;
    for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
        sum += values[idx] * x[col_indicies[idx]];
    }
    y[row] = sum;
}

std::vector<double> simple_gpu_spmv(const CSR_Matrix &h_matrix, const std::vector<float> &h_x, std::vector<float> &y,
                                    int warmup, int repeat) {

    y.resize(h_matrix.rows);

    std::vector<double> elapsed_times;

    int *d_row_offsets = nullptr;
    int *d_col_indices = nullptr;
    float *d_values = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_row_offsets, sizeof(int) * (h_matrix.row_offsets.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_col_indices, sizeof(int) * (h_matrix.col_indices.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_values, sizeof(float) * (h_matrix.values.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_x, sizeof(float) * (h_x.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_y, sizeof(float) * (h_matrix.rows)));

    CUDA_CHECK(cudaMemcpy(d_row_offsets, h_matrix.row_offsets.data(), sizeof(int) * (h_matrix.row_offsets.size()),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, h_matrix.col_indices.data(), sizeof(int) * (h_matrix.col_indices.size()),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_values, h_matrix.values.data(), sizeof(float) * (h_matrix.values.size()), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float) * (h_x.size()), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup + repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        constexpr int threads_per_block = 256;
        int blocks = (h_matrix.rows + threads_per_block - 1) / threads_per_block;

        simple_spmv_kernel<<<blocks, threads_per_block>>>(h_matrix.rows, d_row_offsets, d_col_indices, d_values, d_x,
                                                          d_y);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        if (i >= warmup) {
            elapsed_times.push_back(elapsed_ms);
        }
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, sizeof(float) * (y.size()), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return elapsed_times;
}
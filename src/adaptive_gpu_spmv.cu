#include "cuda_utils.cuh"
#include <cuda_runtime.h>

#include "types.hpp"
#include <vector>

__global__ void csr_normal_rows_kernel(int n_active_rows, const int *rows, const int *row_offsets,
                                       const int *col_indices, const float *values, const float *x, float *y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active_rows)
        return;

    int row = rows[tid];
    float sum = 0.0f;

    for (int jj = row_offsets[row]; jj < row_offsets[row + 1]; ++jj) {
        sum += values[jj] * x[col_indices[jj]];
    }

    y[row] = sum;
}

__global__ void csr_long_rows_kernel(int n_long_rows, const int *long_rows, const int *row_offsets,
                                     const int *col_indices, const float *values, const float *x, float *y) {
    extern __shared__ float shared[];

    int list_id = blockIdx.x;
    if (list_id >= n_long_rows)
        return;

    int row = long_rows[list_id];
    int start = row_offsets[row];
    int end = row_offsets[row + 1];

    float local_sum = 0.0f;

    for (int jj = start + threadIdx.x; jj < end; jj += blockDim.x) {
        local_sum += values[jj] * x[col_indices[jj]];
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = shared[0];
    }
}

std::vector<double> adaptive_gpu_spmv(const CSR_Matrix &h_matrix, const std::vector<float> &h_x, std::vector<float> &y,
                                      int warmup, int repeat) {
    y.resize(h_matrix.rows);

    std::vector<double> elapsed_times;

    std::vector<int> normal_rows;
    std::vector<int> long_rows;

    for (int row = 0; row < h_matrix.rows; ++row) {
        int nnz_row = h_matrix.row_offsets[row + 1] - h_matrix.row_offsets[row];
        if (nnz_row <= 256) {
            normal_rows.push_back(row);
        } else {
            long_rows.push_back(row);
        }
    }

    int *d_normal_rows = nullptr;
    int *d_long_rows = nullptr;

    int *d_row_offsets = nullptr;
    int *d_col_indices = nullptr;
    float *d_values = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    if (!normal_rows.empty()) {
        CUDA_CHECK(cudaMalloc((void **)&d_normal_rows, sizeof(int) * (normal_rows.size())));
    }
    if (!long_rows.empty()) {
        CUDA_CHECK(cudaMalloc((void **)&d_long_rows, sizeof(int) * (long_rows.size())));
    }

    CUDA_CHECK(cudaMalloc((void **)&d_row_offsets, sizeof(int) * (h_matrix.row_offsets.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_col_indices, sizeof(int) * (h_matrix.col_indices.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_values, sizeof(float) * (h_matrix.values.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_x, sizeof(float) * (h_x.size())));
    CUDA_CHECK(cudaMalloc((void **)&d_y, sizeof(float) * (h_matrix.rows)));

    if (!normal_rows.empty()) {
        CUDA_CHECK(
            cudaMemcpy(d_normal_rows, normal_rows.data(), sizeof(int) * (normal_rows.size()), cudaMemcpyHostToDevice));
    }
    if (!long_rows.empty()) {
        CUDA_CHECK(cudaMemcpy(d_long_rows, long_rows.data(), sizeof(int) * (long_rows.size()), cudaMemcpyHostToDevice));
    }
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
        int blocks = (normal_rows.size() + threads_per_block - 1) / threads_per_block;

        if (!normal_rows.empty()) {
            csr_normal_rows_kernel<<<blocks, threads_per_block>>>(normal_rows.size(), d_normal_rows, d_row_offsets,
                                                                  d_col_indices, d_values, d_x, d_y);
        }

        constexpr int long_threads_per_block = 256;
        int long_blocks = long_rows.size();
        size_t shared_bytes = long_threads_per_block * sizeof(float);

        if (!long_rows.empty()) {
            csr_long_rows_kernel<<<long_blocks, long_threads_per_block, shared_bytes>>>(
                long_rows.size(), d_long_rows, d_row_offsets, d_col_indices, d_values, d_x, d_y);
        }
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

    if (d_normal_rows != nullptr) {
        CUDA_CHECK(cudaFree(d_normal_rows));
    }

    if (d_long_rows != nullptr) {
        CUDA_CHECK(cudaFree(d_long_rows));
    }
    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return elapsed_times;
}

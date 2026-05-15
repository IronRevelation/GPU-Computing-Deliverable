// cuSPARSE descriptor/setup sequence adapted from NVIDIA CUDA Library Samples:
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spmv_csr
// Original samples distributed under the Apache License 2.0.

#include "cuda_utils.cuh"
#include "types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cusparse.h>

std::vector<double> cusparse_gpu_spmv(const CSR_Matrix &h_matrix, const std::vector<float> &h_x, std::vector<float> &y,
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

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    CUSPARSE_CHECK(cusparseCreate(&handle));

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, h_matrix.rows, h_matrix.cols, h_matrix.nnz(), d_row_offsets, d_col_indices,
                                     d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, h_matrix.cols, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, h_matrix.rows, d_y, CUDA_R_32F));

    float alpha = 1.0f;
    float beta = 0.0f;

    size_t buffer_size = 0;
    void *d_buffer = nullptr;

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                                           CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &buffer_size));

    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup + repeat; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                                    CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, d_buffer));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        if (i >= warmup) {
            elapsed_times.push_back(elapsed_ms);
        }
    }

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, sizeof(float) * y.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (d_buffer != nullptr) {
        CUDA_CHECK(cudaFree(d_buffer));
    }

    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaFree(d_row_offsets));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return elapsed_times;
}
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cusparse.h>

#define CUDA_CHECK(expr_to_check)                                                                                      \
    do {                                                                                                               \
        cudaError_t result = expr_to_check;                                                                            \
        if (result != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result,                         \
                    cudaGetErrorString(result));                                                                       \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUSPARSE_CHECK(expr)                                                                                           \
    do {                                                                                                               \
        cusparseStatus_t status = (expr);                                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                                       \
            std::fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, status);                         \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)
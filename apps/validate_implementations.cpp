#include "cpu_spmv.hpp"
#include "gpu_spmv.hpp"
#include "matrix_reader.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
    std::vector<std::string> files = input_files(argc, argv);
    std::vector<std::string> errors;

    bool valid = true;
    std::cout << "errors:\n";
    for (const std::string &file : files) {
        CSR_Matrix matrix = read_matrix(file);
        std::vector<float> x = generate_random_vector(matrix.cols, 0);
        std::vector<float> expected = cpu_spmv(matrix, x);

        std::vector<float> simple_results;
        simple_gpu_spmv(matrix, x, simple_results);
        std::vector<float> adaptive_results;
        adaptive_gpu_spmv(matrix, x, adaptive_results);
        std::vector<float> cusparse_results;
        cusparse_gpu_spmv(matrix, x, cusparse_results);

        if (!float_vector_equal(expected, simple_results)) {

            std::cout << "  " << file << "\n";
            std::cout << "  simple: max abs diff: " << max_abs_difference(expected, simple_results) << "\n";
            std::cout << "  tolerance: " << validation_tolerance(expected) << "\n";
            valid = false;
        }

        if (!float_vector_equal(expected, adaptive_results)) {

            std::cout << "  " << file << "\n";
            std::cout << "   adaptive: max abs diff: " << max_abs_difference(expected, adaptive_results) << "\n";
            std::cout << "  tolerance: " << validation_tolerance(expected) << "\n";
            valid = false;
        }
        if (!float_vector_equal(expected, cusparse_results)) {
            std::cout << "  " << file << "\n";
            std::cout << "  cusparse: max abs diff: " << max_abs_difference(expected, cusparse_results) << "\n";
            std::cout << "  tolerance: " << validation_tolerance(expected) << "\n";
            valid = false;
        }
    }

    std::cout << "valid: " << valid << "\n";

    return 0;
}

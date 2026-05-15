#include "cpu_spmv.hpp"
#include "gpu_spmv.hpp"
#include "matrix_reader.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "write_bench.hpp"
#include <filesystem>
#include <vector>

void print_timings(const std::vector<double> &timings) {
    for (double timing : timings) {
        std::cout << timing << "ms ";
    }
}

const int warmup = 10;
const int repeat = 100;

int main(int argc, char **argv) {
    std::vector<std::string> files = input_files(argc, argv);

    for (const std::string &file : files) {
        CSR_Matrix matrix = read_matrix(file);
        std::vector<float> x = generate_random_vector(matrix.cols, 0);

        std::vector<double> cpu_timings = cpu_spmv_timed(matrix, x, warmup, repeat);
        std::vector<float> simple_results;
        std::vector<double> simple_timings = simple_gpu_spmv(matrix, x, simple_results, warmup, repeat);
        std::vector<float> adaptive_results;
        std::vector<double> adaptive_timings = adaptive_gpu_spmv(matrix, x, adaptive_results, warmup, repeat);
        std::vector<float> cusparse_results;
        std::vector<double> cusparse_timings = cusparse_gpu_spmv(matrix, x, cusparse_results, warmup, repeat);

        std::vector<float> expected = cpu_spmv(matrix, x);

        if (!float_vector_equal(expected, simple_results)) {
            print_fatal_error("simple validation failed for " + file);
        }
        if (!float_vector_equal(expected, adaptive_results)) {
            print_fatal_error("adaptive validation failed for " + file);
        }
        if (!float_vector_equal(expected, cusparse_results)) {
            print_fatal_error("cusparse validation failed for " + file);
        }

        std::filesystem::create_directories("results");

        std::filesystem::path input_path(file);
        std::string filename_copy = input_path.stem().string();
        std::filesystem::path output_path = std::filesystem::path("results") / (input_path.stem().string() + ".csv");

        write_benchmark_header(output_path.string());
        write_benchmark(output_path.string(), filename_copy, "cpu", matrix, warmup, repeat, cpu_timings);
        write_benchmark(output_path.string(), filename_copy, "simple", matrix, warmup, repeat, simple_timings);
        write_benchmark(output_path.string(), filename_copy, "adaptive", matrix, warmup, repeat, adaptive_timings);
        write_benchmark(output_path.string(), filename_copy, "cusparse", matrix, warmup, repeat, cusparse_timings);
    }
    return 0;
}

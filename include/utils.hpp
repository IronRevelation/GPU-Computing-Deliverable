#pragma once

#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

void print_fatal_error(std::string message) {
    std::cerr << message << std::endl;
    exit(1);
}

std::vector<float> generate_random_vector(int size, int seed) {
    std::vector<float> vector(size);
    vector.resize(size);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (int i = 0; i < size; ++i) {
        vector[i] = dist(gen);
    }
    return vector;
}

std::vector<std::string> input_files(int argc, char **argv) {
    std::vector<std::string> files;

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            files.push_back(argv[i]);
        }
    } else {
        for (const auto &entry : std::filesystem::directory_iterator("data")) {
            if (entry.path().extension() == ".mtx") {
                files.push_back(entry.path().string());
            }
        }
    }

    return files;
}

double max_abs_difference(const std::vector<float> &a, const std::vector<float> &b) {
    double max_difference = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_difference = std::max(max_difference, (double)(std::abs(a[i] - b[i])));
    }
    return max_difference;
}

double max_abs_value(const std::vector<float> &values) {
    double max_value = 0.0;
    for (float value : values) {
        max_value = std::max(max_value, (double)(std::abs(value)));
    }
    return max_value;
}

double validation_tolerance(const std::vector<float> &reference) {
    constexpr double abs_tolerance = 1.0e-4;
    constexpr double rel_tolerance = 1.0e-3;
    return abs_tolerance + rel_tolerance * max_abs_value(reference);
}

bool float_vector_equal(const std::vector<float> &reference, const std::vector<float> &actual) {
    if (reference.size() != actual.size()) {
        return false;
    }
    return max_abs_difference(reference, actual) <= validation_tolerance(reference);
}

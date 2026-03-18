#ifndef EDGEAI_CPP_UTILS_H
#define EDGEAI_CPP_UTILS_H

#include <vector>
#include <string>
#include <iostream>

namespace edgeai_cpp {

// Utility function to print a vector
template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// Utility function to convert image (e.g., OpenCV Mat) to float vector for TFLite input
// (Placeholder - actual implementation would depend on image library used)
std::vector<float> image_to_float_vector(const void* image_data, int width, int height, int channels);

} // namespace edgeai_cpp

#endif // EDGEAI_CPP_UTILS_H

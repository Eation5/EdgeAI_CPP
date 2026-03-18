# EdgeAI_CPP

![C++](https://img.shields.io/badge/C%2B%2B-GCC%209%2B-blue?style=flat-square&logo=c%2B%2B)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-2.x-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/github/license/Eation5/EdgeAI_CPP?style=flat-square)

## Overview

EdgeAI_CPP is a C++ library designed for deploying and running AI models on edge devices with limited computational resources. It focuses on optimizing inference for computer vision and machine learning tasks using frameworks like TensorFlow Lite and OpenCV. This project provides efficient C++ implementations for common AI workloads, enabling high-performance, low-latency AI applications in embedded systems, IoT devices, and other edge environments.

## Features

- **Optimized Inference**: Leverage TensorFlow Lite for efficient model execution on edge hardware.
- **Computer Vision Primitives**: Integration with OpenCV for image processing, camera interfacing, and visualization.
- **Cross-Platform Compatibility**: Designed to run on various Linux-based embedded systems.
- **Low-Latency Design**: Focus on minimizing inference time and resource consumption.
- **Modular Components**: Easily integrate different pre-trained models or custom-trained models.
- **Example Applications**: Includes ready-to-use examples for common edge AI scenarios.

## Installation

To build and install EdgeAI_CPP, you will need a C++17 compatible compiler (e.g., GCC 9+), CMake, TensorFlow Lite, and OpenCV. 

```bash
git clone https://github.com/Eation5/EdgeAI_CPP.git
cd EdgeAI_CPP
mkdir build
cd build
cmake ..
make
# sudo make install # Optional: to install to system-wide paths
```

## Usage

Here's a simple example demonstrating how to load a TensorFlow Lite model and perform inference:

```cpp
#include "edgeai_cpp/tflite_model.h"
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    // Assume a dummy model path for demonstration
    std::string model_path = "path/to/your/model.tflite"; 
    
    // In a real scenario, you would have a valid .tflite model file
    // For this example, we will simulate model loading and inference.
    
    // Create a dummy input tensor (e.g., for an image classification model expecting 1x224x224x3 float input)
    std::vector<float> input_data(1 * 224 * 224 * 3, 0.5f); // Example: 1 image, 224x224, 3 channels, all 0.5
    
    // Simulate loading and inference
    std::cout << "Simulating TensorFlow Lite model loading and inference..." << std::endl;
    
    // Simulate output data (e.g., 10 classes output)
    std::vector<float> output_data(10);
    // Fill with some dummy probabilities
    std::iota(output_data.begin(), output_data.end(), 0.1f); 

    std::cout << "Inference successful. Output probabilities:" << std::endl;
    for (float val : output_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

## Project Structure

```
EdgeAI_CPP/
├── README.md
├── CMakeLists.txt
├── edgeai_cpp/
│   ├── include/
│   │   ├── edgeai_cpp/
│   │   │   ├── tflite_model.h
│   │   │   └── utils.h
│   └── src/
│       ├── tflite_model.cpp
│       └── utils.cpp
└── examples/
    └── tflite_inference.cpp
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to get started.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries, please open an issue on GitHub or contact Matthew Wilson at [matthew.wilson.ai@example.com](mailto:matthew.wilson.ai@example.com).

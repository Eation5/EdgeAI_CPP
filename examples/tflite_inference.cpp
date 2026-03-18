#include "edgeai_cpp/tflite_model.h"
#include "edgeai_cpp/utils.h"
#include <iostream>
#include <vector>
#include <numeric>

int main() {
    // In a real scenario, replace with your actual .tflite model path
    std::string model_path = "./model.tflite"; 
    
    edgeai_cpp::TFLiteModel model(model_path);

    // Simulate model loading. In a real case, ensure model.tflite exists.
    // For this example, we will bypass actual loading and simulate inference.
    // if (!model.LoadModel()) {
    //     std::cerr << "Failed to load TFLite model." << std::endl;
    //     return 1;
    // }

    // Create a dummy input tensor (e.g., for an image classification model expecting 1x224x224x3 float input)
    // This should match the input shape of your TFLite model
    std::vector<float> input_data(1 * 224 * 224 * 3, 0.5f); // Example: 1 image, 224x224, 3 channels, all 0.5
    
    std::vector<float> output_data;

    std::cout << "Simulating TensorFlow Lite model inference..." << std::endl;
    
    // Simulate inference by filling output_data directly
    // In a real scenario, you would call model.RunInference(input_data, output_data);
    output_data.resize(10); // Assuming 10 output classes
    std::iota(output_data.begin(), output_data.end(), 0.1f); // Fill with some dummy probabilities

    std::cout << "Inference successful. Output probabilities:" << std::endl;
    edgeai_cpp::print_vector(output_data, "Model Output");

    return 0;
}

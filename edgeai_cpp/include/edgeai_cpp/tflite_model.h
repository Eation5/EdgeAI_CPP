#ifndef EDGEAI_CPP_TFLITE_MODEL_H
#define EDGEAI_CPP_TFLITE_MODEL_H

#include <string>
#include <vector>
#include <memory>

// Forward declaration for TensorFlow Lite interpreter
namespace tflite {
    class Interpreter;
    class FlatBufferModel;
}

namespace edgeai_cpp {

class TFLiteModel {
public:
    TFLiteModel(const std::string& model_path);
    ~TFLiteModel();

    // Load the model and initialize the interpreter
    bool LoadModel();

    // Run inference
    // input_data: flattened input tensor data
    // output_data: vector to store flattened output tensor data
    bool RunInference(const std::vector<float>& input_data, std::vector<float>& output_data);

    // Get input and output tensor details (e.g., shape, type)
    // For simplicity, this example won't implement full tensor details retrieval
    // but a real application would need this.

private:
    std::string model_path_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    bool model_loaded_ = false;
};

} // namespace edgeai_cpp

#endif // EDGEAI_CPP_TFLITE_MODEL_H

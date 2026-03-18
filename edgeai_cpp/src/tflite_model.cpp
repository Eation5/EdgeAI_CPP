#include "edgeai_cpp/tflite_model.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

namespace edgeai_cpp {

TFLiteModel::TFLiteModel(const std::string& model_path)
    : model_path_(model_path) {}

TFLiteModel::~TFLiteModel() = default;

bool TFLiteModel::LoadModel() {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
        std::cerr << "Failed to load model: " << model_path_ << std::endl;
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);

    if (!interpreter_) {
        std::cerr << "Failed to construct interpreter." << std::endl;
        return false;
    }

    interpreter_->AllocateTensors();
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return false;
    }

    // Optional: Print model details
    // tflite::PrintInterpreterState(interpreter_.get());

    model_loaded_ = true;
    return true;
}

bool TFLiteModel::RunInference(const std::vector<float>& input_data, std::vector<float>& output_data) {
    if (!model_loaded_) {
        std::cerr << "Model not loaded. Call LoadModel() first." << std::endl;
        return false;
    }

    if (interpreter_->inputs().size() != 1) {
        std::cerr << "Model must have exactly one input tensor for this example." << std::endl;
        return false;
    }

    int input_tensor_idx = interpreter_->inputs()[0];
    TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_idx);

    if (input_tensor->bytes != input_data.size() * sizeof(float)) {
        std::cerr << "Input data size mismatch. Expected " << input_tensor->bytes << " bytes, got " << input_data.size() * sizeof(float) << " bytes." << std::endl;
        return false;
    }

    std::copy(input_data.begin(), input_data.end(), interpreter_->typed_tensor<float>(input_tensor_idx));

    if (interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter." << std::endl;
        return false;
    }

    if (interpreter_->outputs().size() != 1) {
        std::cerr << "Model must have exactly one output tensor for this example." << std::endl;
        return false;
    }

    int output_tensor_idx = interpreter_->outputs()[0];
    TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx);

    output_data.resize(output_tensor->bytes / sizeof(float));
    std::copy(interpreter_->typed_tensor<float>(output_tensor_idx), 
              interpreter_->typed_tensor<float>(output_tensor_idx) + output_data.size(), 
              output_data.begin());

    return true;
}

} // namespace edgeai_cpp

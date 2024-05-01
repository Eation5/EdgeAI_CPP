#include "edgeai_cpp/tflite_model.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h> // Optional: for NNAPI delegate

namespace edgeai_cpp {

TFLiteModel::TFLiteModel(const std::string& model_path)
    : model_path_(model_path), interpreter_initialized_(false) {
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_) {
        throw std::runtime_error("Failed to load TFLite model from: " + model_path_);
    }
}

TFLiteModel::~TFLiteModel() = default;

void TFLiteModel::InitInterpreter() {
    if (interpreter_initialized_) {
        std::cerr << "Interpreter already initialized." << std::endl;
        return;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);

    // Optional: Add delegates for acceleration (e.g., NNAPI, GPU)
    // TfLiteDelegate* nnapi_delegate = tflite::NnApiDelegate();
    // builder.AddDelegate(nnapi_delegate);

    if (builder(&interpreter_) != kTfLiteOk) {
        throw std::runtime_error("Failed to construct TFLite interpreter.");
    }

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors for TFLite interpreter.");
    }

    interpreter_initialized_ = true;
    std::cout << "TFLite interpreter initialized successfully for model: " << model_path_ << std::endl;
    // tflite::PrintInterpreterState(interpreter_.get()); // For debugging
}

void TFLiteModel::RunInference(const std::vector<std::vector<float>>& input_data,
                               std::vector<std::vector<float>>& output_data) {
    if (!interpreter_initialized_) {
        throw std::runtime_error("Interpreter not initialized. Call InitInterpreter() first.");
    }

    if (input_data.size() != interpreter_->inputs().size()) {
        throw std::runtime_error("Input data vector count mismatch. Expected " + 
                                 std::to_string(interpreter_->inputs().size()) + 
                                 ", got " + std::to_string(input_data.size()) + ".");
    }

    // Copy input data to input tensors
    for (size_t i = 0; i < input_data.size(); ++i) {
        int input_tensor_idx = interpreter_->inputs()[i];
        TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_idx);

        if (input_tensor->bytes != input_data[i].size() * sizeof(float)) {
            throw std::runtime_error("Input tensor " + std::to_string(i) + ": data size mismatch. Expected " + 
                                     std::to_string(input_tensor->bytes) + " bytes, got " + 
                                     std::to_string(input_data[i].size() * sizeof(float)) + " bytes.");
        }
        std::copy(input_data[i].begin(), input_data[i].end(), interpreter_->typed_tensor<float>(input_tensor_idx));
    }

    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        throw std::runtime_error("Failed to invoke TFLite interpreter.");
    }

    // Copy output data from output tensors
    output_data.clear();
    for (size_t i = 0; i < interpreter_->outputs().size(); ++i) {
        int output_tensor_idx = interpreter_->outputs()[i];
        TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_idx);

        std::vector<float> current_output(output_tensor->bytes / sizeof(float));
        std::copy(interpreter_->typed_tensor<float>(output_tensor_idx),
                  interpreter_->typed_tensor<float>(output_tensor_idx) + current_output.size(),
                  current_output.begin());
        output_data.push_back(current_output);
    }
}

TensorInfo TFLiteModel::getTensorInfo(int tensor_idx) const {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_idx);
    TensorInfo info;
    info.name = tensor->name ? tensor->name : "";
    info.type = tensor->type;
    info.bytes = tensor->bytes;
    for (int i = 0; i < tensor->dims->size; ++i) {
        info.shape.push_back(tensor->dims->data[i]);
    }
    return info;
}

std::vector<TensorInfo> TFLiteModel::GetInputTensorInfo() const {
    if (!interpreter_initialized_) {
        throw std::runtime_error("Interpreter not initialized. Call InitInterpreter() first.");
    }
    std::vector<TensorInfo> input_info;
    for (int idx : interpreter_->inputs()) {
        input_info.push_back(getTensorInfo(idx));
    }
    return input_info;
}

std::vector<TensorInfo> TFLiteModel::GetOutputTensorInfo() const {
    if (!interpreter_initialized_) {
        throw std::runtime_error("Interpreter not initialized. Call InitInterpreter() first.");
    }
    std::vector<TensorInfo> output_info;
    for (int idx : interpreter_->outputs()) {
        output_info.push_back(getTensorInfo(idx));
    }
    return output_info;
}

} // namespace edgeai_cpp

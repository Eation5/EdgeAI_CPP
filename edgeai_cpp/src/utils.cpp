#include "edgeai_cpp/utils.h"
#include <stdexcept>

namespace edgeai_cpp {

cv::Mat load_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Convert BGR to RGB
    return image;
}

cv::Mat resize_image(const cv::Mat& image, int target_width, int target_height) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);
    return resized_image;
}

cv::Mat normalize_image(const cv::Mat& image) {
    cv::Mat normalized_image;
    image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);
    return normalized_image;
}

std::vector<float> image_to_float_vector(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    // Ensure image is float32 and 3 channels (RGB)
    cv::Mat float_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, float_image, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, float_image, cv::COLOR_RGBA2RGB);
    }
    else {
        float_image = image.clone();
    }
    float_image.convertTo(float_image, CV_32F);

    std::vector<float> float_vector;
    float_vector.reserve(float_image.total() * float_image.channels());

    // TFLite models often expect channels last (HWC) and flattened
    // Iterate through rows, then columns, then channels
    for (int i = 0; i < float_image.rows; ++i) {
        for (int j = 0; j < float_image.cols; ++j) {
            for (int c = 0; c < float_image.channels(); ++c) {
                float_vector.push_back(float_image.at<cv::Vec3f>(i, j)[c]);
            }
        }
    }
    return float_vector;
}

cv::Mat normalize_image_mean_std(const cv::Mat& image, 
                                 const std::vector<float>& mean, 
                                 const std::vector<float>& std) {
    if (image.empty()) {
        return image;
    }
    if (mean.size() != image.channels() || std.size() != image.channels()) {
        throw std::runtime_error("Mean and Std vectors must match image channels.");
    }

    cv::Mat float_image;
    image.convertTo(float_image, CV_32F);

    std::vector<cv::Mat> channels(image.channels());
    cv::split(float_image, channels);

    for (int i = 0; i < image.channels(); ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::Mat normalized_image;
    cv::merge(channels, normalized_image);
    return normalized_image;
}

cv::Mat float_vector_to_image(const std::vector<float>& data, int width, int height, int channels, bool normalize_to_255) {
    if (data.size() != static_cast<size_t>(width * height * channels)) {
        throw std::runtime_error("Data size does not match specified image dimensions.");
    }

    cv::Mat image(height, width, CV_32FC(channels));
    size_t k = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                image.at<cv::Vec3f>(i, j)[c] = data[k++];
            }
        }
    }

    if (normalize_to_255) {
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8U);
    } else {
        image.convertTo(image, CV_8U);
    }
    
    if (channels == 3) {
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR); // Convert back to BGR for OpenCV display/save
    }

    return image;
}

} // namespace edgeai_cpp

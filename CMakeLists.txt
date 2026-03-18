cmake_minimum_required(VERSION 3.10)
project(EdgeAI_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find TensorFlow Lite
# You might need to adjust this path based on your TensorFlow Lite installation
# For example, if installed via apt-get, headers might be in /usr/include/tensorflow/lite
# and libraries in /usr/lib
find_path(TFLITE_INCLUDE_DIR tensorflow/lite/version.h
          PATHS /usr/local/include /usr/include
          PATH_SUFFIXES tensorflow/lite)
find_library(TFLITE_LIB tensorflow-lite
             PATHS /usr/local/lib /usr/lib)

if(TFLITE_INCLUDE_DIR AND TFLITE_LIB)
    message(STATUS "Found TensorFlow Lite: ${TFLITE_INCLUDE_DIR}")
    include_directories(${TFLITE_INCLUDE_DIR})
    link_directories(${TFLITE_LIB})
else()
    message(FATAL_ERROR "TensorFlow Lite not found. Please install it or set TFLITE_INCLUDE_DIR and TFLITE_LIB.")
endif()

# Define include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/edgeai_cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

# Add library sources
add_library(edgeai_cpp_lib
    edgeai_cpp/src/tflite_model.cpp
    edgeai_cpp/src/utils.cpp
)

# Link libraries
target_link_libraries(edgeai_cpp_lib
    ${OpenCV_LIBS}
    ${TFLITE_LIB}
)

# Add example executable
add_executable(tflite_inference examples/tflite_inference.cpp)
target_link_libraries(tflite_inference edgeai_cpp_lib)

# Install rules (optional)
install(TARGETS edgeai_cpp_lib tflite_inference
        DESTINATION bin)
install(DIRECTORY edgeai_cpp/include/edgeai_cpp/
        DESTINATION include)

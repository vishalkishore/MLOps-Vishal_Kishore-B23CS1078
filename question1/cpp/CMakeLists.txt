cmake_minimum_required(VERSION 3.18)
project(question1_onnx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-<version> when configuring CMake.")
endif()

find_package(nlohmann_json REQUIRED)

add_executable(question1_onnx_cpp main.cpp)

target_include_directories(
  question1_onnx_cpp
  PRIVATE
    ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(
  question1_onnx_cpp
  PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(
  question1_onnx_cpp
  PRIVATE
    onnxruntime
    nlohmann_json::nlohmann_json
)

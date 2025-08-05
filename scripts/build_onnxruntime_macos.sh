#!/bin/bash

# Build ONNX Runtime for macOS ARM64
# This script builds ONNX Runtime from source to get the required dylib

set -e

# Create build directory
mkdir -p build
cd build

# Clone ONNX Runtime if not already present
if [ ! -d "onnxruntime" ]; then
  echo "Cloning ONNX Runtime..."
  git clone --recursive https://github.com/Microsoft/onnxruntime
fi

cd onnxruntime

# Build for macOS ARM64
echo "Building ONNX Runtime for macOS ARM64..."
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests

# Copy the resulting dylib to our third_party directory
echo "Copying dylib to third_party directory..."
mkdir -p ../../third_party
cp build/MacOS/Release/libonnxruntime.dylib ../../third_party/onnxruntime_arm64.dylib

echo "✅ ONNX Runtime library built successfully!"
echo "Library location: $(pwd)/../../third_party/onnxruntime_arm64.dylib"

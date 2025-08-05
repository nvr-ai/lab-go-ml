# Scripts Directory

## build_onnxruntime_macos.sh

Builds ONNX Runtime from source for macOS ARM64 (Apple Silicon).

### Usage

```bash
chmod +x scripts/build_onnxruntime_macos.sh
./scripts/build_onnxruntime_macos.sh
```

### What it does


1. Creates a `build` directory
2. Clones the Microsoft ONNX Runtime repository
3. Builds the shared library for macOS ARM64
4. Copies the resulting `libonnxruntime.dylib` to `third_party/onnxruntime_arm64.dylib`

### Prerequisites

* macOS with Apple Silicon (M1/M2/M3)
* Xcode Command Line Tools: `xcode-select --install`
* CMake: Install via Homebrew `brew install cmake` or download from https://cmake.org
* Python 3.x (for the build process)
* Git with LFS support

### Build Time

The build process can take 30-60 minutes depending on your system.

### Troubleshooting

**Build fails with missing dependencies:**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake via Homebrew (if available)
brew install cmake

# Or download CMake from https://cmake.org
```

**Out of disk space:**
The build requires several GB of space. Clean up or use a different drive.

**Build succeeds but library not found:**
Check that the file exists at `third_party/onnxruntime_arm64.dylib` and has the correct permissions.

### Alternative

If building from source is not feasible, consider:


1. Using GoCV's DNN module for ONNX inference
2. Converting ONNX models to other formats (TensorFlow Lite, Core ML)
3. Using Python subprocess calls to leverage pip-installed onnxruntime
4. Cloud-based inference APIs



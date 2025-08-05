# ONNX Runtime on macOS ARM64 (Apple Silicon)

## The Problem

Microsoft does not provide pre-built ONNX Runtime C++ libraries for macOS ARM64 (Apple Silicon). The official releases only include:
- Python packages (pip)
- Java packages (Maven)  
- Mobile packages (CocoaPods)
- **But NOT standalone C++ dylib files**

## Solutions

### Option 1: Build from Source (Recommended)

Build ONNX Runtime from source to get the required dylib:

```bash
# Run the build script
chmod +x scripts/build_onnxruntime_macos.sh
./scripts/build_onnxruntime_macos.sh
```

This will:
1. Clone the ONNX Runtime repository
2. Build it for macOS ARM64  
3. Copy the resulting `libonnxruntime.dylib` to `third_party/onnxruntime_arm64.dylib`

**Prerequisites:**
- Xcode Command Line Tools
- CMake
- Python 3.x

### Option 2: Disable ONNX Runtime (Fallback)

If you can't build from source, you can disable ONNX Runtime:

```go
config := Config{
    ModelPath:           "path/to/model.onnx",
    InputShape:          image.Point{X: 416, Y: 416},
    ConfidenceThreshold: 0.5,
    NMSThreshold:        0.5,
    RelevantClasses:     []string{"person", "car"},
    DisableONNXRuntime:  true, // This will skip ONNX Runtime
}

// This will return an error explaining ONNX Runtime is disabled
session, err := NewSession(config)
```

### Option 3: Use Alternative Libraries

Consider using pure Go alternatives like:
- `gorgonia.org/gorgonia` (already in your dependencies)
- `github.com/owulveryck/onnx-go` 
- OpenCV DNN module via GoCV (already in your dependencies)

## Testing

The tests will automatically skip if the ONNX Runtime library is not found:

```bash
go test ./onnx/...
```

Output:
```
=== RUN   TestNewONNXDetector
    onnx_test.go:28: Skipping ONNX Runtime test - library not available: ONNX Runtime library not found at ../third_party/onnxruntime_arm64.dylib
--- SKIP: TestNewONNXDetector (0.00s)
```

## Why This Happens

1. **Microsoft's Focus**: ONNX Runtime's main distribution is via package managers (pip, Maven, NuGet)
2. **Build Complexity**: C++ libraries require complex build processes for each platform
3. **Size Concerns**: C++ libraries are large and platform-specific
4. **Mobile Priority**: For macOS, Microsoft focuses on mobile/iOS packages rather than desktop C++ libraries

## Alternative Approaches

If you must use ONNX models without building from source:

1. **Use Python subprocess**: Call Python with onnxruntime from Go
2. **Use GoCV DNN**: Load ONNX models with OpenCV's DNN module
3. **Convert to other formats**: Convert ONNX to TensorFlow Lite or Core ML
4. **Use cloud APIs**: Run inference via cloud services 
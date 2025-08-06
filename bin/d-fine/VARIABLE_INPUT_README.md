# D-FINE ONNX Export with Variable Input Dimensions

This directory contains the enhanced D-FINE ONNX export system that supports variable input dimensions for object detection with onnxruntime-go integration.

## Features

- **Variable Input Dimensions**: Support for dynamic width and height inputs
- **Aspect Ratio Support**: 16:9 and 4:3 aspect ratios
- **Resolution Constraints**: Configurable maximum dimensions
- **Performance Optimized**: Uses ONNX opset 17 for optimal dynamic shape support
- **Go Integration**: Full compatibility with onnxruntime-go
- **Comprehensive Testing**: Built-in validation and testing framework

## Supported Resolutions

### 16:9 Aspect Ratio
- 160×90
- 320×180
- 640×360 (nHD)
- 960×540 (qHD 540p)

### 4:3 Aspect Ratio
- 160×120
- 320×240
- 640×480
- 960×720

## Quick Start

### 1. Setup Environment

```bash
./cli setup
```

### 2. Basic Export

```bash
# Export with default settings (both aspect ratios, all resolutions)
./cli export --resume path/to/model.pth

# Export for specific aspect ratio
./cli export --resume path/to/model.pth --aspect-ratios 16:9

# Export with dimension constraints
./cli export --resume path/to/model.pth --max-width 640 --max-height 480
```

### 3. Advanced Export with Testing

```bash
# Export with full validation and testing
./cli export --resume path/to/model.pth --test-inference --validate-resolutions
```

## Command Line Interface

### Export Options

| Option | Description | Default |
|--------|-------------|---------|
| `--aspect-ratios` | Supported aspect ratios (`16:9`, `4:3`) | Both |
| `--max-width` | Maximum width constraint | None |
| `--max-height` | Maximum height constraint | None |
| `--no-dynamic-shapes` | Disable dynamic input shapes | Enabled |
| `--test-inference` | Test inference with different resolutions | Disabled |
| `--validate-resolutions` | Validate model with all resolutions | Disabled |

### Examples

```bash
# Export for 16:9 aspect ratio only
./cli export --resume model.pth --aspect-ratios 16:9

# Export with maximum 640×360 constraint
./cli export --aspect-ratios 16:9 --max-width 640 --max-height 360

# Export with full validation
./cli export --resume model.pth --test-inference --validate-resolutions

# Export with fixed dimensions (no dynamic shapes)
./cli export --resume model.pth --no-dynamic-shapes
```

## Python API

### Basic Usage

```python
from export import export_d_fine_model

# Export with default settings
model_path = export_d_fine_model(
    config_path="configs/dfine/dfine_hgnetv2_l_coco.yml",
    resume_path="path/to/model.pth"
)
```

### Advanced Configuration

```python
# Export with specific resolutions
model_path = export_d_fine_model(
    config_path="configs/dfine/dfine_hgnetv2_l_coco.yml",
    resume_path="path/to/model.pth",
    aspect_ratios=["16:9"],
    max_dimensions=(640, 360),
    enable_dynamic_shapes=True
)
```

### Resolution Configuration

```python
from export import ResolutionConfig, Resolution

# Get all supported resolutions
resolutions = ResolutionConfig.get_supported_resolutions()

# Get resolutions for specific aspect ratio
resolutions_16_9 = ResolutionConfig.get_supported_resolutions(["16:9"])

# Find optimal resolution for target dimensions
optimal = ResolutionConfig.get_optimal_resolution(1000, 600, ["16:9"])
print(f"Optimal resolution: {optimal}")  # 960x540 (16:9)

# Validate a resolution
is_valid = ResolutionConfig.validate_resolution(640, 360)
print(f"640x360 is valid: {is_valid}")  # True
```

## Go Integration

### Setup

```bash
go mod init your-project
go get github.com/yalue/onnxruntime_go
```

### Basic Usage

```go
package main

import (
    "log"
    "github.com/yalue/onnxruntime_go"
)

func main() {
    // Initialize ONNX Runtime
    err := onnxruntime_go.InitializeEnvironment()
    if err != nil {
        log.Fatal(err)
    }
    defer onnxruntime_go.DestroyEnvironment()
    
    // Create session
    session, err := onnxruntime_go.NewSession("model.onnx", onnxruntime_go.NewSessionOptions())
    if err != nil {
        log.Fatal(err)
    }
    defer session.Destroy()
    
    // Create input tensors (example for 640x360)
    batchSize := 1
    channels := 3
    height := 360
    width := 640
    
    // Image tensor (NCHW format)
    imageShape := []int64{int64(batchSize), int64(channels), int64(height), int64(width)}
    imageData := make([]float32, batchSize*channels*height*width)
    // ... fill imageData with your image data ...
    
    imageTensor, err := onnxruntime_go.NewTensor(imageShape, imageData)
    if err != nil {
        log.Fatal(err)
    }
    defer imageTensor.Destroy()
    
    // Size tensor
    sizeShape := []int64{int64(batchSize), 2}
    sizeData := []float32{float32(width), float32(height)}
    
    sizeTensor, err := onnxruntime_go.NewTensor(sizeShape, sizeData)
    if err != nil {
        log.Fatal(err)
    }
    defer sizeTensor.Destroy()
    
    // Run inference
    inputs := []onnxruntime_go.Value{imageTensor, sizeTensor}
    outputs, err := session.Run(inputs)
    if err != nil {
        log.Fatal(err)
    }
    
    // Process outputs: labels, boxes, scores
    // ... handle detection results ...
    
    // Clean up
    for _, output := range outputs {
        output.Destroy()
    }
}
```

### Testing Go Integration

```bash
# Run the Go integration test
go run test_go_integration.go path/to/exported/model.onnx
```

## Testing

### Python Tests

```bash
# Run unit tests
python test_export.py

# Run integration tests (requires D-FINE setup)
python test_export.py --integration
```

### Go Tests

```bash
# Test with exported model
go run test_go_integration.go model.onnx
```

## Performance Considerations

### Dynamic vs Fixed Shapes

- **Dynamic Shapes (Default)**: Supports variable input dimensions but may have slight overhead
- **Fixed Shapes**: Better performance but requires separate models for different resolutions

### Memory Usage

Resolution memory usage (approximate):
- 160×90: ~0.2MB per batch
- 320×180: ~0.7MB per batch  
- 640×360: ~2.8MB per batch
- 960×540: ~6.2MB per batch

### Optimization Tips

1. Use the smallest resolution that meets your accuracy requirements
2. Consider fixed shapes for high-throughput scenarios
3. Enable model simplification for production deployments
4. Use appropriate batch sizes based on available memory

## Troubleshooting

### Common Issues

**Export fails with ONNX version error:**
```bash
# Ensure you have compatible ONNX version
uv pip install onnx>=1.12.0 onnxsim>=0.4.0
```

**Go integration fails:**
```bash
# Ensure ONNX Runtime C++ libraries are installed
# On macOS: brew install onnxruntime
# On Ubuntu: Follow onnxruntime installation guide
```

**Model validation fails:**
```bash
# Check if model was exported with dynamic shapes
./cli export --resume model.pth --validate-resolutions
```

### Debug Mode

Enable verbose output for debugging:

```bash
# Python
python export.py --resume model.pth --config configs/dfine/dfine_hgnetv2_l_coco.yml

# CLI with validation
./cli export --resume model.pth --test-inference --validate-resolutions
```

## File Structure

```
bin/d-fine/
├── cli                      # Main CLI script
├── export.py               # Enhanced export script with variable dimensions
├── test_export.py          # Python test suite
├── test_go_integration.go  # Go integration test
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── repo/                  # D-FINE repository (auto-cloned)
```

## License

This implementation follows the same license as the D-FINE project.
# Model Exporting

## ONNX

### Capabilities

- **Variable Input Dimensions**: Dynamic width/height support for inference
- **Aspect Ratio Support**: Full support for 16:9 and 4:3 aspect ratios
- **Resolution Configuration**: 8 supported resolutions (160-960px widths)
- **Performance Optimization**: ONNX opset 17 with dynamic shape optimizations
- **Go Integration**: Full compatibility with onnxruntime-go

### Supported Resolutions

- **16:9**: 160×90, 320×180, 640×360, 960×540
- **4:3**: 160×120, 320×240, 640×480, 960×720

### Implementation Details

#### `export.py` - Complete rewrite with

- **Resolution configuration system** (ResolutionConfig, Resolution classes).
- **Dynamic ONNX export** with variable input dimensions.
- **Comprehensive validation** and testing functions.
- **Enhanced CLI** with new resolution parameters.

#### `cli` - Updated bash script with

- **New resolution parameter support**.
- **Enhanced help documentation**.
- **Backward compatibility**.

#### `test_export.py` - Comprehensive test suite

- **Unit tests** for resolution configuration.
- **Mock export functionality** testing.
- **Integration test** framework.

#### `test_go_integration.go` - Go compatibility testing

- **Full onnxruntime-go integration** validation.
- **Multi-resolution inference** testing.
- **Performance** validation.

#### `VARIABLE_INPUT_README.md` - Complete documentation

- **API usage examples**.
- **CLI reference**.
- **Go integration** guide.
- **Performance** considerations.

### CLI Usage

#### Export with default settings (both aspect ratios)

./cli export --resume model.pth

#### Export for 16:9 only with dimension constraints  

./cli export --aspect-ratios 16:9 --max-width 640 --max-height 360

#### Export with full validation and testing

./cli export --resume model.pth --test-inference --validate-resolutions

#### Export with fixed dimensions (no dynamic shapes)

./cli export --resume model.pth --no-dynamic-shapes

### Performance & Accuracy Optimized

- Uses ONNX opset 17 for best dynamic shape support.
- Implements memory-efficient tensor operations.
- Supports model simplification for production.
- Configurable batch processing capabilities.
- Resolution-aware optimization strategies.

### Export D-FINE to ONNX (defaults)

```sh
./bin/d-fine/cli export
```

### Export D-FINE to ONNX (custom)

```sh
./bin/d-fine/cli export --config configs/dfine/dfine_hgnetv2_l_coco.yml --resume path/to/model.pth
```

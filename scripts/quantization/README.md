# ONNX Model Quantization Pipeline

This directory contains a comprehensive quantization pipeline for ONNX models, supporting both FP16 and INT8 quantization with accuracy preservation validation.

## Directory Structure

```
scripts/quantization/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── quantize_fp16.py            # FP16 quantization for GPU/CPU
├── quantize_int8.py            # INT8 quantization with calibration
├── validate_quantization.py    # Accuracy validation pipeline
├── cli.py                      # Command-line interface
├── calibration_data/           # Calibration dataset management
│   ├── __init__.py
│   ├── dataset_loader.py       # Dataset loading utilities
│   └── data_generator.py       # Representative data generation
├── models/                     # Model-specific quantization configs
│   ├── __init__.py
│   ├── yolo_config.py          # YOLO family configurations
│   ├── dfine_config.py         # D-FINE configurations
│   └── common_config.py        # Shared configurations
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── onnx_utils.py           # ONNX model utilities
│   ├── metrics.py              # Accuracy metrics calculation
│   └── profiling.py            # Performance profiling
└── validation/                 # Validation and testing
    ├── __init__.py
    ├── accuracy_tests.py       # Accuracy validation tests
    └── performance_tests.py    # Performance benchmarks
```

## Features

### FP16 Quantization (GPU/CPU Optimization)
- Automatic FP16 conversion using ONNX Runtime quantization toolkit
- GPU-optimized inference with CUDA/TensorRT providers
- CPU compatibility with accelerated execution
- Minimal accuracy loss (typically <1%)

### INT8 Quantization (CPU Optimization)
- Calibration-based quantization using representative datasets
- Advanced calibration methods (MinMax, Entropy, Percentile)
- CPU-optimized inference with DNNL provider
- Target: 50% resource reduction while preserving ≥98% accuracy

### Accuracy Preservation
- Automated validation against PyTorch baselines
- IoU-based accuracy measurement (IoU ≥ 0.5)
- mAP preservation validation (≥98% requirement)
- Multi-resolution accuracy comparison
- Comprehensive test coverage on 50+ diverse sample frames

### Performance Optimization
- Quantized model performance profiling
- Latency measurement and comparison
- Memory usage optimization
- Throughput analysis

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# FP16 quantization
python quantize_fp16.py --model path/to/model.onnx --output path/to/model_fp16.onnx

# INT8 quantization with calibration
python quantize_int8.py --model path/to/model.onnx --output path/to/model_int8.onnx --calibration-data path/to/data/

# Validate quantized model
python validate_quantization.py --original path/to/model.onnx --quantized path/to/model_int8.onnx --test-data path/to/test/

# CLI interface
python cli.py --help
```

### Advanced Configuration

```bash
# Custom calibration method
python quantize_int8.py --model model.onnx --calibration-method entropy --samples 1000

# Multi-resolution validation
python validate_quantization.py --original model.onnx --quantized model_int8.onnx --resolutions 320,640,1024

# Performance profiling
python utils/profiling.py --model model_int8.onnx --provider cpu --iterations 100
```

## Requirements

- Python 3.8+
- ONNX Runtime 1.16+
- PyTorch 2.0+
- NumPy
- Pillow
- tqdm
- matplotlib (for visualizations)

## Model Support

### Currently Supported
- D-FINE (all variants)
- YOLO family (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11)
- Faster R-CNN (ResNet50/101, FPN variants)
- RT-DETR (Real-time detection transformers)

### Adding New Models
1. Create configuration in `models/your_model_config.py`
2. Implement preprocessing pipeline
3. Add validation metrics
4. Test with representative data

## Performance Targets

### FP16 Quantization
- Target: Minimal accuracy loss (<1%)
- GPU inference speedup: 1.5-2x
- Memory reduction: ~50%
- Latency: ≤30ms@640×640 on GPU

### INT8 Quantization
- Target: ≥98% accuracy preservation
- CPU inference speedup: 2-4x
- Memory reduction: ~75%
- Resource usage: 50% reduction

## Validation Metrics

### Accuracy Metrics
- **mAP@IoU[0.5:0.95]**: Primary accuracy metric
- **IoU ≥ 0.5**: Detection fidelity threshold
- **Class-specific accuracy**: Per-class performance analysis
- **Small object detection**: Performance on objects <32²pixels

### Performance Metrics
- **Latency**: Mean/P95 inference time
- **Throughput**: FPS at different resolutions
- **Memory usage**: Peak/average memory consumption
- **Resource utilization**: CPU/GPU usage patterns

## Integration with Go Runtime

Quantized models integrate seamlessly with the Go inference pipeline:

```go
// Load quantized model
config := onnx.DefaultConfig()
config.ModelPath = "model_int8.onnx"
config.OptimizationConfig.ExecutionProviders = []onnx.ExecutionProviderConfig{
    {Provider: onnx.DNNLExecutionProvider, Enabled: true, Priority: 10},
    {Provider: onnx.CPUExecutionProvider, Enabled: true, Priority: 1},
}

session, err := onnx.NewSession(config)
```

## Troubleshooting

### Common Issues
- **ORT_INVALID_GRAPH**: Model contains unsupported operators
- **Calibration failures**: Insufficient or non-representative calibration data
- **Accuracy degradation**: Requires model-specific calibration tuning

### Solutions
- Use ONNX opset 17+ for best quantization support
- Ensure calibration data represents deployment distribution
- Tune quantization parameters per model architecture
- Validate on diverse test scenarios

## Contributing

1. Add new quantization methods in respective modules
2. Ensure comprehensive testing with validation pipeline
3. Update configuration templates for new models
4. Document performance characteristics and limitations
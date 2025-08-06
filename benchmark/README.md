# ML Inference Benchmarking Framework

A comprehensive, reusable benchmarking framework for ML inference performance testing. Supports multiple models, image formats, resolutions, and provides detailed performance metrics with filesystem persistence.

## Features

- **Multi-format Support**: JPEG, WebP, PNG image formats
- **Resolution Testing**: Compare performance across different input resolutions
- **Model Comparison**: Support for YOLO, D-Fine, and other ONNX models
- **Comprehensive Metrics**: FPS, memory usage, CPU stats, disk I/O, detection counts
- **Flexible Configuration**: JSON-based configuration and scenario management
- **Results Persistence**: JSON and CSV output formats
- **CLI Tool**: Standalone command-line interface for easy benchmarking
- **Extensible Design**: Plugin-based architecture for different inference engines

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "time"
    
    "github.com/nvr-ai/go-ml/benchmark"
)

func main() {
    // Create benchmark suite
    engine := benchmark.NewONNXEngine()
    suite := benchmark.NewBenchmarkSuite(engine, "./results")
    
    // Load test images
    suite.LoadTestImages("./test_images", benchmark.FormatJPEG)
    
    // Create and add scenarios
    scenario := benchmark.NewScenarioBuilder("yolo_416_jpeg").
        WithModel(benchmark.ModelYOLO, "./models/yolov8n.onnx").
        WithResolution(416, 416).
        WithImageFormat(benchmark.FormatJPEG).
        WithIterations(100).
        Build()
    
    suite.AddScenario(scenario)
    
    // Run benchmarks
    ctx := context.WithTimeout(context.Background(), 10*time.Minute)
    suite.RunAllScenarios(ctx)
}
```

### Using CLI Tool

```bash
# Quick benchmark
go run ./benchmark/cmd/benchmark -images ./test_images -model ./yolov8n.onnx -quick

# Resolution comparison
go run ./benchmark/cmd/benchmark -images ./test_images -model ./yolov8n.onnx -resolutions

# Format comparison
go run ./benchmark/cmd/benchmark -images ./test_images -model ./yolov8n.onnx -formats

# Comprehensive benchmark
go run ./benchmark/cmd/benchmark -images ./test_images -model ./yolov8n.onnx -comprehensive

# Using configuration file
go run ./benchmark/cmd/benchmark -config ./config.json -scenarios ./scenarios.json
```

## Configuration

### Benchmark Configuration

```json
{
  "output_dir": "./benchmark_results",
  "test_images_path": "./test_images",
  "model_paths": {
    "yolo": "./models/yolov8n.onnx",
    "dfine": "./models/d-fine.onnx"
  },
  "max_concurrency": 1,
  "timeout_seconds": 3600,
  "save_detailed_log": true
}
```

### Custom Scenarios

```json
{
  "name": "Custom Performance Test",
  "description": "Custom benchmark scenarios",
  "scenarios": [
    {
      "name": "yolo_416_jpeg_fast",
      "model_type": "yolo",
      "model_path": "./models/yolov8n.onnx",
      "resolution": {"width": 416, "height": 416, "name": "416x416"},
      "image_format": "jpeg",
      "batch_size": 1,
      "iterations": 50,
      "warmup_runs": 5
    }
  ]
}
```

## Scenario Types

### Predefined Scenarios

- **Quick Scenarios**: Fast test with common configurations
- **Comprehensive Scenarios**: All combinations of models, resolutions, and formats
- **Resolution Comparison**: Different input resolutions with same model
- **Format Comparison**: Different image formats with same model and resolution
- **Model Comparison**: Different models with same configuration

### Custom Scenarios

Use the `ScenarioBuilder` for flexible scenario creation:

```go
scenario := benchmark.NewScenarioBuilder("custom_test").
    WithModel(benchmark.ModelYOLO, "./model.onnx").
    WithResolution(640, 640).
    WithImageFormat(benchmark.FormatWebP).
    WithIterations(100).
    WithWarmupRuns(10).
    WithBatchSize(1).
    Build()
```

## Performance Metrics

### Collected Metrics

- **Timing**: Total duration, resize duration, inference duration
- **Throughput**: Frames per second (FPS)
- **Memory**: Allocation, heap usage, GC statistics
- **Detection**: Count of detected objects, error rates
- **System**: CPU usage, disk I/O statistics

### Output Formats

#### JSON (Detailed Results)
```json
{
  "scenario": {...},
  "timestamp": "2024-01-01T12:00:00Z",
  "total_duration": 5000000000,
  "frames_per_second": 45.2,
  "memory_stats": {
    "alloc_bytes": 104857600,
    "total_alloc_bytes": 524288000,
    "num_gc": 12
  },
  "detection_count": 1250,
  "error_rate": 0.02
}
```

#### CSV (Summary)
```csv
Scenario,Model,Resolution,Format,FPS,Total_Duration_ms,Avg_Memory_MB,Detections,Error_Rate
yolo_416_jpeg,yolo,416x416,jpeg,45.20,5000.00,100.00,1250,0.02
```

## Architecture

### Core Components

- **BenchmarkSuite**: Main orchestrator for benchmark execution
- **InferenceEngine**: Interface for different ML inference backends
- **ScenarioBuilder**: Fluent API for scenario creation
- **PerformanceMetrics**: Comprehensive metrics collection
- **ResultsPersistence**: JSON and CSV output management

### Extensibility

The framework uses an interface-based design for easy extension:

```go
type InferenceEngine interface {
    LoadModel(modelPath string, config map[string]interface{}) error
    Predict(ctx context.Context, img image.Image) (interface{}, error)
    Close() error
    GetModelInfo() map[string]interface{}
}
```

## Integration with ONNX Tests

The framework seamlessly integrates with existing ONNX tests:

```go
// In your *_test.go files
func BenchmarkONNXInference(b *testing.B) {
    suite := benchmark.NewBenchmarkSuite(benchmark.NewONNXEngine(), "./results")
    suite.LoadTestImages("./test_images", benchmark.FormatJPEG)
    
    // Add scenarios and run
    suite.RunAllScenarios(context.Background())
}
```

## Example Results

```
=== BENCHMARK RESULTS SUMMARY ===
Total scenarios: 6
Results saved to: ./benchmark_results

  yolo_416_jpeg: 45.20 FPS (98.50 MB memory)
  yolo_416_webp: 32.15 FPS (156.20 MB memory)
  yolo_416_png: 38.75 FPS (205.10 MB memory)
  yolo_640_jpeg: 28.90 FPS (185.30 MB memory)
  yolo_1024_jpeg: 12.45 FPS (412.80 MB memory)

Best performing scenario: yolo_416_jpeg (45.20 FPS)
```

## Best Practices

1. **Use appropriate iterations**: 100+ for stable results, 10-50 for quick tests
2. **Include warmup runs**: 5-10 runs to stabilize performance
3. **Test multiple formats**: JPEG typically fastest, WebP best compression
4. **Monitor memory usage**: Watch for memory leaks in long-running tests
5. **Save configurations**: Use JSON files for reproducible benchmarks
6. **Analyze trends**: Compare results across different model versions

## Future Enhancements

- Support for additional inference backends (TensorRT, CoreML)
- Distributed benchmarking across multiple machines
- Real-time monitoring and alerting
- Integration with CI/CD pipelines
- Web-based results dashboard
- Automated performance regression detection
# ML + `go` Lab

## Commands

| Command                           | Description                             |
| --------------------------------- | --------------------------------------- |
| `make tidy`                       | Tidy up the go.mod file                 |
| `make gocv/webcam`                | Run the gocv webcam test                |
| `make gorgonia/tiny-yolo-v3-coco` | Run the gorgonia tiny-yolo-v3-coco test |

## Testing

```sh
../go-ml 🌱 main [✘!?] ➜ make test
go test ./...
2025/08/03 18:27:18.769394 integration_test.go:51: 🚀 Motion Detection Test Suite Starting...
2025/08/03 18:27:18.769630 integration_test.go:52: 📊 Configuration: regression=true, verbose=true, report=true
2025/08/03 18:27:18.769634 integration_test.go:54: 💾 Results directory: ./test_results
2025/08/03 18:27:18.769637 integration_test.go:55: ⚡ Benchmark directory: ./benchmarks
2025/08/03 18:27:18.769639 integration_test.go:56: 🖥️  Runtime: 10 CPUs available
2025/08/03 18:27:19.086823 integration_test.go:81: ✅ Test suite completed in 317.148042ms with exit code 0
2025/08/03 18:27:19.086842 integration_test.go:87: 🔍 Starting regression analysis...
2025/08/03 18:27:19.086845 integration_test.go:155:   Analyzing TestMotionSegmenterCreation...
2025/08/03 18:27:19.087955 integration_test.go:168:   ❌ REGRESSION in TestMotionSegmenterCreation: REGRESSION: Duration increased by 460.2% (threshold: 10.0%); Consistent degradation trend: 55.6% per execution
2025/08/03 18:27:19.087962 integration_test.go:172:      Previous mean: 0.60 ms
2025/08/03 18:27:19.087965 integration_test.go:173:      Previous P95:  1.75 ms
2025/08/03 18:27:19.087967 integration_test.go:174:      Trend: 55.6% per execution
2025/08/03 18:27:19.087969 integration_test.go:179:      💡 Review recent code changes for performance impacts
2025/08/03 18:27:19.087971 integration_test.go:179:      💡 Run profiling to identify bottlenecks
2025/08/03 18:27:19.087973 integration_test.go:179:      💡 Consider reverting recent changes if regression is severe
2025/08/03 18:27:19.087974 integration_test.go:179:      💡 High variance detected (CV=1.11), consider stabilizing test environment
2025/08/03 18:27:19.087976 integration_test.go:155:   Analyzing TestSubtractBackground...
2025/08/03 18:27:19.088159 integration_test.go:184:   ✅ STABLE: TestSubtractBackground
2025/08/03 18:27:19.088164 integration_test.go:155:   Analyzing TestSegmentMotionPipeline...
2025/08/03 18:27:19.088955 integration_test.go:184:   ✅ STABLE: TestSegmentMotionPipeline
2025/08/03 18:27:19.088964 integration_test.go:155:   Analyzing TestIdempotency...
2025/08/03 18:27:19.089849 integration_test.go:184:   ✅ STABLE: TestIdempotency
2025/08/03 18:27:19.089854 integration_test.go:94: ❌ REGRESSIONS DETECTED - Build should fail!
2025/08/03 18:27:19.089858 integration_test.go:104: 📝 Generating test report...
2025/08/03 18:27:19.089995 integration_test.go:113: 📊 Report generated: test_results/reports/test_report_20250803_182719.html
FAIL    github.com/nvr-ai/go-ml/test    0.745s
ok      github.com/nvr-ai/go-ml/tmp/gorgonia-yolov3-testing/tiny-yolo-v3-coco   0.438s [no tests to run]
```

### Webcam Test

```javascript
make gocv/webcam
```

## Test Coverage

- [`BenchmarkSubtractBackground`](../motion/detector.go) - Background subtraction only.
- [`BenchmarkFullPipeline`](../motion/detector.go) - Full motion segmentation pipeline.

## Benchmark Coverage

The benchmark suite provides comprehensive coverage for performance and regression testing for the motion detection system using GoCV.

### Motion Detection Component Benchmarks (motion/detector.go)

- [`BenchmarkMotionDetectorProcess`](../test/benchmarks_test.go) - Core motion processing logic (185ns/op).
- [`BenchmarkMotionDetectorFPS`](../test/benchmarks_test.go) - FPS calculation performance.
- [`BenchmarkMotionDetectorMetrics`](../test/benchmarks_test.go) - Metrics collection overhead.

### Motion Segmentation Component Benchmarks (images/motion.go)

- [`BenchmarkApplyThreshold`](../test/benchmarks_test.go) - Thresholding operation (51μs/op).
- [`BenchmarkFillGaps`](../test/benchmarks_test.go) - Morphological operations.
- [`BenchmarkDetectContours`](../test/benchmarks_test.go) - Contour detection performance.

### Integration & Advanced Benchmarks

- [`BenchmarkIntegratedMotionDetection`](../test/benchmarks_test.go) - End-to-end pipeline (11.37ms/op).
- [`BenchmarkMultiResolution`](../test/benchmarks_test.go) - Performance across 480p/720p/1080p/4K.
- [`BenchmarkMemoryAllocation`](../test/benchmarks_test.go) - Memory allocation patterns.
- [`BenchmarkConcurrentProcessing`](../test/benchmarks_test.go) - Multi-threaded performance.

### Stress & Edge Case Benchmarks

- [`BenchmarkLongRunningDetection`](../test/benchmarks_test.go) - Memory stability over time.
- [`BenchmarkHighMotionScenario`](../test/benchmarks_test.go) - Multiple motion regions (21.55ms/op).
- [`BenchmarkNoiseResilience`](../test/benchmarks_test.go) - Performance with noisy input.
- [`BenchmarkRapidSceneChanges`](../test/benchmarks_test.go) - Scene transition handling.
- [`BenchmarkProfilerOverhead`](../test/benchmarks_test.go) - Profiling impact measurement.
- [`BenchmarkEdgeCaseFrames`](../test/benchmarks_test.go) - Edge cases (black/white/gradient frames).

### Key Features

1. Result Persistence: All benchmarks save detailed JSON results with metadata. (e.g. `results/motion_detector_benchmark_results.json`).
2. Memory Tracking: Heap allocation monitoring and GC cycle tracking.
3. Performance Metrics: `ns/op`, `allocations/op`, `FPS` potential calculations.
4. Realistic Test Data: Mock frame generator for deterministic testing (black/white/gradient frames).
5. Resource Management: Proper `gocv.Mat` cleanup to prevent memory leaks.
6. `gocv` Integration: Fixed API compatibility issues for proper OpenCV operations (e.g. `Mat.Close()`).

### Performance Insights

- Motion Detection Logic: 185ns per process operation (very fast).
- Thresholding: 51μs per operation (1920x1080).
- End-to-End Pipeline: 11.37ms per frame (88 FPS potential).
- High Motion Stress: 21.55ms per frame (46 FPS under stress).

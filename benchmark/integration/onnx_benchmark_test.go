package integration

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/benchmark"
	"github.com/nvr-ai/go-ml/benchmark/engines"
	"github.com/nvr-ai/go-ml/images"
)

// BenchmarkONNXInference runs comprehensive benchmarks using the new framework
func BenchmarkONNXInference(b *testing.B) {
	// Skip if ONNX runtime not available
	engine := engines.NewONNXEngine()
	testConfig := map[string]interface{}{
		"input_shape":          []int{416, 416},
		"confidence_threshold": 0.5,
		"nms_threshold":        0.4,
	}

	err := engine.LoadModel("../../data/yolov8n.onnx", testConfig)
	if err != nil {
		if strings.Contains(err.Error(), "ONNX Runtime library not found") {
			b.Skipf("Skipping ONNX benchmark - library not available: %v", err)
			return
		}
		b.Fatalf("Failed to load model: %v", err)
	}
	engine.Close()

	// Create benchmark suite
	suite := benchmark.NewBenchmarkSuite(engines.NewONNXEngine(), "./benchmark_results")

	// Load test images
	err = suite.LoadTestImages("../../../../ml/corpus/images/videos/freeway-view-22-seconds-1080p.mp4", images.FormatJPEG)
	if err != nil {
		b.Logf("Warning: Could not load test images from corpus, using fallback: %v", err)
		// Try to create a simple test image if corpus not available
		err = suite.LoadTestImages("../test_images", images.FormatJPEG)
		if err != nil {
			b.Skipf("No test images available: %v", err)
			return
		}
	}

	// Define model paths
	modelPaths := map[benchmark.ModelType]string{
		benchmark.ModelYOLO: "../../data/yolov8n.onnx",
	}

	// Generate benchmark scenarios
	predefined := &benchmark.PredefinedScenarios{}

	// Add quick scenarios for benchmarking
	quickScenarios := predefined.GetQuickScenarios(modelPaths)
	for _, scenario := range quickScenarios.Scenarios {
		// Reduce iterations for benchmark
		scenario.Iterations = 10
		scenario.WarmupRuns = 2
		suite.AddScenario(scenario)
	}

	// Run benchmarks
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	b.ResetTimer()
	err = suite.RunAllScenarios(ctx)
	if err != nil {
		b.Fatalf("Benchmark failed: %v", err)
	}

	// Print summary
	results := suite.GetResults()
	for _, result := range results {
		b.Logf("Scenario: %s, FPS: %.2f, Memory: %.2f MB",
			result.Scenario.Name,
			result.FramesPerSecond,
			float64(result.MemoryStats.AllocBytes)/(1024*1024))
	}
}

// BenchmarkResolutionComparison benchmarks different input resolutions
func BenchmarkResolutionComparison(b *testing.B) {
	engine := engines.NewONNXEngine()
	testConfig := map[string]interface{}{
		"input_shape":          []int{416, 416},
		"confidence_threshold": 0.5,
		"nms_threshold":        0.4,
	}

	err := engine.LoadModel("../../data/yolov8n.onnx", testConfig)
	if err != nil {
		if strings.Contains(err.Error(), "ONNX Runtime library not found") {
			b.Skipf("Skipping resolution benchmark - library not available: %v", err)
			return
		}
		b.Fatalf("Failed to load model: %v", err)
	}
	engine.Close()

	suite := benchmark.NewBenchmarkSuite(engines.NewONNXEngine(), "./benchmark_results")

	err = suite.LoadTestImages("../../../../ml/corpus/images/videos/freeway-view-22-seconds-1080p.mp4", images.FormatJPEG)
	if err != nil {
		b.Skipf("No test images available: %v", err)
		return
	}

	// Test different resolutions
	predefined := &benchmark.PredefinedScenarios{}
	resolutionScenarios := predefined.GetResolutionComparisonScenarios(benchmark.ModelYOLO, "../../data/yolov8n.onnx")

	for _, scenario := range resolutionScenarios.Scenarios {
		scenario.Iterations = 5 // Reduce for benchmark
		scenario.WarmupRuns = 1
		suite.AddScenario(scenario)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	b.ResetTimer()
	err = suite.RunAllScenarios(ctx)
	if err != nil {
		b.Fatalf("Resolution benchmark failed: %v", err)
	}
}

// BenchmarkFormatComparison benchmarks different image formats
func BenchmarkFormatComparison(b *testing.B) {
	engine := engines.NewONNXEngine()
	testConfig := map[string]interface{}{
		"input_shape":          []int{416, 416},
		"confidence_threshold": 0.5,
		"nms_threshold":        0.4,
	}

	err := engine.LoadModel("../../data/yolov8n.onnx", testConfig)
	if err != nil {
		if strings.Contains(err.Error(), "ONNX Runtime library not found") {
			b.Skipf("Skipping format benchmark - library not available: %v", err)
			return
		}
		b.Fatalf("Failed to load model: %v", err)
	}
	engine.Close()

	suite := benchmark.NewBenchmarkSuite(engines.NewONNXEngine(), "./benchmark_results")

	// Test with different formats (we'll load the same images but process them as different formats)
	formats := []images.ImageFormat{images.FormatJPEG, images.FormatWebP, images.FormatPNG}

	for _, format := range formats {
		err = suite.LoadTestImages("../../../../ml/corpus/images/videos/freeway-view-22-seconds-1080p.mp4", format)
		if err != nil {
			continue
		}

		predefined := &benchmark.PredefinedScenarios{}
		resolution := benchmark.Resolution{Width: 416, Height: 416, Name: "416x416"}
		formatScenarios := predefined.GetFormatComparisonScenarios(benchmark.ModelYOLO, "../../data/yolov8n.onnx", resolution)

		for _, scenario := range formatScenarios.Scenarios {
			if scenario.ImageFormat == format {
				scenario.Iterations = 5
				scenario.WarmupRuns = 1
				suite.AddScenario(scenario)
				break
			}
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	b.ResetTimer()
	err = suite.RunAllScenarios(ctx)
	if err != nil {
		b.Fatalf("Format benchmark failed: %v", err)
	}
}

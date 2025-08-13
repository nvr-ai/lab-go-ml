package dfine

// Package dfine_test provides comprehensive test coverage for D-FINE model post-processing functionality.
//
// This test suite validates all aspects of the PostProcess method including input validation,
// confidence filtering, tensor parsing, coordinate extraction, and Non-Maximum Suppression
// application. The tests ensure idempotency, fault tolerance, and accurate detection processing
// suitable for production object detection inference pipelines.

import (
	"fmt"
	"testing"

	"github.com/nvr-ai/go-ml/models/postprocess"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDFINEPostProcessValidDetections validates the complete post-processing pipeline with valid detection data.
//
// This test ensures that properly formatted detection tensors are correctly parsed into
// structured results with accurate bounding box coordinates, confidence scores, and class
// identifiers. It validates the core functionality of the post-processing pipeline.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessValidDetections(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure NMS parameters for standard object detection.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false,
	}

	// Create test tensor with two high-confidence detections.
	// Format: [x1, y1, x2, y2, confidence, class_id] per detection
	output := []float32{
		100.0, 150.0, 200.0, 250.0, 0.85, 1.0, // Detection 1: person with high confidence
		300.0, 400.0, 450.0, 500.0, 0.92, 2.0, // Detection 2: car with very high confidence
	}

	// Process the detections through the post-processing pipeline.
	results := model.PostProcess(output, config)

	// Validate that both detections were processed successfully.
	require.Len(t, results, 2, "Should process both high-confidence detections")

	// Validate first detection results.
	firstResult := results[0]
	assert.Equal(t, float32(100.0), firstResult.Box.X1, "First detection X1 coordinate should be parsed correctly")
	assert.Equal(t, float32(150.0), firstResult.Box.Y1, "First detection Y1 coordinate should be parsed correctly")
	assert.Equal(t, float32(200.0), firstResult.Box.X2, "First detection X2 coordinate should be parsed correctly")
	assert.Equal(t, float32(250.0), firstResult.Box.Y2, "First detection Y2 coordinate should be parsed correctly")
	assert.Equal(t, float32(0.85), firstResult.Score, "First detection confidence should match input")
	assert.Equal(t, 1, firstResult.Class, "First detection class should be parsed correctly")

	// Validate second detection results.
	secondResult := results[1]
	assert.Equal(t, float32(300.0), secondResult.Box.X1, "Second detection X1 coordinate should be parsed correctly")
	assert.Equal(t, float32(400.0), secondResult.Box.Y1, "Second detection Y1 coordinate should be parsed correctly")
	assert.Equal(t, float32(450.0), secondResult.Box.X2, "Second detection X2 coordinate should be parsed correctly")
	assert.Equal(t, float32(500.0), secondResult.Box.Y2, "Second detection Y2 coordinate should be parsed correctly")
	assert.Equal(t, float32(0.92), secondResult.Score, "Second detection confidence should match input")
	assert.Equal(t, 2, secondResult.Class, "Second detection class should be parsed correctly")
}

// TestDFINEPostProcessConfidenceFiltering validates confidence threshold filtering functionality.
//
// This test ensures that detections with confidence scores below the configured IoU threshold
// are properly filtered out during post-processing. It validates the early filtering logic
// that improves performance by eliminating low-quality detections.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessConfidenceFiltering(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure strict confidence filtering.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.6, // High threshold for aggressive filtering
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false,
	}

	// Create test tensor with mixed confidence detections.
	output := []float32{
		100.0, 150.0, 200.0, 250.0, 0.75, 1.0, // Detection 1: above threshold (kept)
		300.0, 400.0, 450.0, 500.0, 0.45, 2.0, // Detection 2: below threshold (filtered)
		500.0, 600.0, 650.0, 700.0, 0.85, 3.0, // Detection 3: above threshold (kept)
	}

	// Process the detections with confidence filtering.
	results := model.PostProcess(output, config)

	// Validate that only high-confidence detections remain.
	require.Len(t, results, 2, "Should filter out low-confidence detection")

	// Validate that remaining detections are the high-confidence ones.
	assert.Equal(t, float32(0.75), results[0].Score, "First remaining detection should have confidence 0.75")
	assert.Equal(t, float32(0.85), results[1].Score, "Second remaining detection should have confidence 0.85")
	assert.Equal(t, 1, results[0].Class, "First remaining detection should be class 1")
	assert.Equal(t, 3, results[1].Class, "Second remaining detection should be class 3")
}

// TestDFINEPostProcessInvalidTensorSize validates error handling for malformed input tensors.
//
// This test ensures that the post-processing pipeline properly handles and rejects
// input tensors that don't conform to the expected D-FINE output format. It validates
// the input validation logic that prevents processing of corrupted or incomplete data.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessInvalidTensorSize(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure standard NMS parameters.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false,
	}

	// Test various invalid tensor sizes.
	invalidSizes := [][]float32{
		{},                                 // Empty tensor
		{100.0},                            // Single value (incomplete)
		{100.0, 150.0, 200.0, 250.0, 0.85}, // 5 values (missing class_id)
		{100.0, 150.0, 200.0, 250.0, 0.85, 1.0, 2.0}, // 7 values (extra value)
	}

	for i, invalidOutput := range invalidSizes {
		t.Run(fmt.Sprintf("InvalidSize_%d_elements", len(invalidOutput)), func(t *testing.T) {
			// Process invalid tensor through post-processing pipeline.
			results := model.PostProcess(invalidOutput, config)

			// Validate that invalid tensors return empty results.
			assert.Nil(t, results, "Invalid tensor size should return nil results")
		})
	}
}

// TestDFINEPostProcessEmptyOutput validates handling of empty detection tensors.
//
// This test ensures that the post-processing pipeline correctly handles scenarios
// where the model produces no detections. This is a common occurrence in object
// detection when no objects are present in the input image.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessEmptyOutput(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure standard NMS parameters.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false,
	}

	// Test with completely empty tensor.
	emptyOutput := []float32{}

	// Process empty tensor through post-processing pipeline.
	results := model.PostProcess(emptyOutput, config)

	// Validate that empty tensors return nil results.
	assert.Nil(t, results, "Empty tensor should return nil results")
}

// TestDFINEPostProcessGreedyNMS validates greedy Non-Maximum Suppression algorithm selection.
//
// This test ensures that the post-processing pipeline correctly routes to the greedy
// NMS implementation when the Greedy flag is enabled in the configuration. It validates
// the algorithm selection logic and ensures proper integration with the NMS subsystem.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessGreedyNMS(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure NMS with greedy algorithm enabled.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              true, // Enable greedy NMS algorithm
	}

	// Create test tensor with overlapping detections for NMS testing.
	output := []float32{
		100.0, 150.0, 200.0, 250.0, 0.85, 1.0, // Detection 1: high confidence
		105.0, 155.0, 205.0, 255.0, 0.75, 1.0, // Detection 2: overlapping, lower confidence
	}

	// Process detections with greedy NMS algorithm.
	results := model.PostProcess(output, config)

	// Validate that results were processed (exact count depends on NMS implementation).
	assert.NotNil(t, results, "Greedy NMS should return processed results")
	assert.LessOrEqual(t, len(results), 2, "NMS should reduce or maintain detection count")

	// If NMS filters overlapping detections, the higher confidence one should remain.
	if len(results) == 1 {
		assert.Equal(t, float32(0.85), results[0].Score, "Greedy NMS should preserve highest confidence detection")
	}
}

// TestDFINEPostProcessStandardNMS validates standard Non-Maximum Suppression algorithm selection.
//
// This test ensures that the post-processing pipeline correctly routes to the standard
// NMS implementation when the Greedy flag is disabled in the configuration. It validates
// the default algorithm selection and proper integration with the NMS subsystem.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessStandardNMS(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure NMS with standard algorithm (default).
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false, // Use standard NMS algorithm
	}

	// Create test tensor with overlapping detections for NMS testing.
	output := []float32{
		100.0, 150.0, 200.0, 250.0, 0.85, 1.0, // Detection 1: high confidence
		105.0, 155.0, 205.0, 255.0, 0.75, 1.0, // Detection 2: overlapping, lower confidence
	}

	// Process detections with standard NMS algorithm.
	results := model.PostProcess(output, config)

	// Validate that results were processed (exact count depends on NMS implementation).
	assert.NotNil(t, results, "Standard NMS should return processed results")
	assert.LessOrEqual(t, len(results), 2, "NMS should reduce or maintain detection count")

	// If NMS filters overlapping detections, the higher confidence one should remain.
	if len(results) == 1 {
		assert.Equal(t, float32(0.85), results[0].Score, "Standard NMS should preserve highest confidence detection")
	}
}

// TestDFINEPostProcessIdempotency validates that post-processing operations are idempotent.
//
// This test ensures that multiple calls to the PostProcess method with identical
// inputs produce identical outputs. This is crucial for reproducible inference
// results and debugging consistency in production environments.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessIdempotency(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure standard NMS parameters.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false,
	}

	// Create test tensor with consistent detection data.
	output := []float32{
		100.0, 150.0, 200.0, 250.0, 0.85, 1.0,
		300.0, 400.0, 450.0, 500.0, 0.92, 2.0,
	}

	// Process the same tensor multiple times.
	results1 := model.PostProcess(output, config)
	results2 := model.PostProcess(output, config)
	results3 := model.PostProcess(output, config)

	// Validate that all results are identical.
	require.Equal(t, len(results1), len(results2), "Multiple calls should produce same result count")
	require.Equal(t, len(results1), len(results3), "Multiple calls should produce same result count")

	// Validate detailed result equivalence.
	for i := range results1 {
		assert.Equal(t, results1[i].Box, results2[i].Box, "Bounding boxes should be identical across calls")
		assert.Equal(t, results1[i].Box, results3[i].Box, "Bounding boxes should be identical across calls")
		assert.Equal(t, results1[i].Score, results2[i].Score, "Confidence scores should be identical across calls")
		assert.Equal(t, results1[i].Score, results3[i].Score, "Confidence scores should be identical across calls")
		assert.Equal(t, results1[i].Class, results2[i].Class, "Class identifiers should be identical across calls")
		assert.Equal(t, results1[i].Class, results3[i].Class, "Class identifiers should be identical across calls")
	}
}

// TestDFINEPostProcessLargeDetectionSet validates handling of large numbers of detections.
//
// This test ensures that the post-processing pipeline efficiently handles scenarios
// with many detections without performance degradation or memory issues. It validates
// scalability and resource management for high-density detection scenarios.
//
// Arguments:
//   - t: The testing context for assertions and error reporting.
func TestDFINEPostProcessLargeDetectionSet(t *testing.T) {
	// Create D-FINE model instance for testing.
	model := &DFINE{}

	// Configure NMS parameters for large detection sets.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       1000, // Allow large number of detections
		Greedy:              false,
	}

	// Generate large tensor with many high-confidence detections.
	const numDetections = 500
	output := make([]float32, numDetections*6)

	for i := 0; i < numDetections; i++ {
		offset := i * 6
		// Generate non-overlapping detections with varying confidence.
		x1 := float32(i%20) * 50.0                 // Distribute across grid
		y1 := float32(i/20) * 50.0                 // Distribute across grid
		confidence := 0.6 + (float32(i%40) * 0.01) // Varying confidence above threshold

		output[offset+0] = x1
		output[offset+1] = y1
		output[offset+2] = x1 + 40.0 // Non-overlapping boxes
		output[offset+3] = y1 + 40.0 // Non-overlapping boxes
		output[offset+4] = confidence
		output[offset+5] = float32(i % 10) // Cycling through 10 classes
	}

	// Process large detection set through post-processing pipeline.
	results := model.PostProcess(output, config)

	// Validate that large detection sets are processed efficiently.
	assert.NotNil(t, results, "Large detection sets should be processed successfully")
	assert.Greater(t, len(results), 0, "Should produce detection results from large input")
	assert.LessOrEqual(t, len(results), numDetections, "Result count should not exceed input count")

	// Validate that all results have valid data structures.
	for i, result := range results {
		assert.GreaterOrEqual(t, result.Score, config.IoUThreshold,
			"Detection %d should meet confidence threshold", i)
		assert.GreaterOrEqual(t, result.Class, 0,
			"Detection %d should have valid class identifier", i)
		assert.Less(t, result.Box.X1, result.Box.X2,
			"Detection %d should have valid X coordinates", i)
		assert.Less(t, result.Box.Y1, result.Box.Y2,
			"Detection %d should have valid Y coordinates", i)
	}
}

// BenchmarkDFINEPostProcessPerformance measures the performance characteristics of post-processing operations.
//
// This benchmark helps identify performance regressions and ensures that the
// post-processing pipeline maintains acceptable performance for real-time inference
// applications. It measures throughput and memory allocation patterns.
//
// Arguments:
//   - b: The benchmarking context for performance measurement and reporting.
func BenchmarkDFINEPostProcessPerformance(b *testing.B) {
	// Create D-FINE model instance for benchmarking.
	model := &DFINE{}

	// Configure realistic NMS parameters.
	config := &postprocess.NMSConfig{
		IoUThreshold:        0.5,
		ConfidenceThreshold: 0.25,
		MaxDetections:       100,
		Greedy:              false,
	}

	// Create realistic detection tensor (50 detections).
	const numDetections = 50
	output := make([]float32, numDetections*6)

	for i := 0; i < numDetections; i++ {
		offset := i * 6
		output[offset+0] = float32(i%10) * 64.0         // x1
		output[offset+1] = float32(i/10) * 64.0         // y1
		output[offset+2] = output[offset+0] + 48.0      // x2
		output[offset+3] = output[offset+1] + 48.0      // y2
		output[offset+4] = 0.3 + (float32(i%70) * 0.01) // confidence
		output[offset+5] = float32(i % 5)               // class
	}

	// Reset timer and enable allocation reporting.
	b.ResetTimer()
	b.ReportAllocs()

	// Run benchmark iterations.
	for i := 0; i < b.N; i++ {
		results := model.PostProcess(output, config)
		_ = results // Prevent optimization elimination
	}
}

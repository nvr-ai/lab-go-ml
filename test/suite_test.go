package test

import (
	"fmt"
	"image"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"gocv.io/x/gocv"
)

// TestMotionSegmenterCreation verifies proper initialization of the MotionSegmenter.
func TestMotionSegmenterCreation(t *testing.T) {
	store := NewTestResultStore("./test_results")
	start := time.Now()

	result := &TestResult{
		TestName:  "TestMotionSegmenterCreation",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	segmenter := images.NewMotionSegmenter()
	if segmenter == nil {
		result.Success = false
		result.Error = "Failed to create MotionSegmenter"
		t.Fatal(result.Error)
	}
	defer segmenter.Close()

	// Verify all components are initialized.
	if segmenter.BackgroundSubtractor == (gocv.BackgroundSubtractorMOG2{}) {
		result.Success = false
		result.Error = "BackgroundSubtractor not initialized"
		t.Fatal(result.Error)
	}
}

// TestSubtractBackground validates background subtraction with deterministic inputs.
func TestSubtractBackground(t *testing.T) {
	store := NewTestResultStore("./test_results")
	generator := NewMockFrameGenerator(640, 480)

	testCases := []struct {
		name        string
		frameFunc   func() gocv.Mat
		expectEmpty bool
	}{
		{
			name:        "static_frame",
			frameFunc:   generator.GenerateStaticFrame,
			expectEmpty: false,
		},
		{
			name: "motion_frame",
			frameFunc: func() gocv.Mat {
				return generator.GenerateMotionFrame(100, 100, 50)
			},
			expectEmpty: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			start := time.Now()
			result := &TestResult{
				TestName:  fmt.Sprintf("TestSubtractBackground_%s", tc.name),
				Timestamp: start,
				Success:   true,
			}
			defer func() {
				result.Duration = time.Since(start)
				store.Save(result)
			}()

			segmenter := images.NewMotionSegmenter()
			defer segmenter.Close()

			frame := tc.frameFunc()
			defer frame.Close()

			result.InputChecksum = images.ComputeMatChecksum(frame)

			err := segmenter.SubtractBackground(frame)
			if err != nil {
				result.Success = false
				result.Error = err.Error()
				t.Errorf("SubtractBackground failed: %v", err)
			}

			result.OutputChecksum = images.ComputeMatChecksum(segmenter.Delta)

			if segmenter.Delta.Empty() && !tc.expectEmpty {
				result.Success = false
				result.Error = "Delta mat is unexpectedly empty"
				t.Error(result.Error)
			}
		})
	}
}

// TestSegmentMotionPipeline validates the complete motion segmentation pipeline.
func TestSegmentMotionPipeline(t *testing.T) {
	store := NewTestResultStore("./test_results")
	generator := NewMockFrameGenerator(640, 480)

	start := time.Now()
	result := &TestResult{
		TestName:  "TestSegmentMotionPipeline",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	segmenter := images.NewMotionSegmenter()
	defer segmenter.Close()

	// Initialize kernel for morphological operations.
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()

	// Process multiple frames to establish background model.
	for i := 0; i < 10; i++ {
		frame := generator.GenerateStaticFrame()
		segmenter.SegmentMotion(frame)
		frame.Close()
	}

	// Introduce motion and detect contours.
	motionFrame := generator.GenerateMotionFrame(200, 200, 100)
	defer motionFrame.Close()

	result.InputChecksum = images.ComputeMatChecksum(motionFrame)

	contours := segmenter.SegmentMotion(motionFrame)
	result.Contours = contours.Size()
	result.OutputChecksum = images.ComputeMatChecksum(segmenter.Threshold)

	if contours.Size() == 0 {
		result.Success = false
		result.Error = "No contours detected when motion was expected"
		t.Error(result.Error)
	}

	t.Logf("Detected %d contours", contours.Size())
}

// TestIdempotency verifies that identical inputs produce identical outputs.
func TestIdempotency(t *testing.T) {
	store := NewTestResultStore("./test_results")
	generator := NewMockFrameGenerator(640, 480)

	start := time.Now()
	result := &TestResult{
		TestName:  "TestIdempotency",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	frame := generator.GenerateMotionFrame(150, 150, 75)
	defer frame.Close()

	checksums := make([]string, 3)

	for i := 0; i < 3; i++ {
		segmenter := images.NewMotionSegmenter()
		segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))

		// Process same sequence each time.
		for j := 0; j < 5; j++ {
			staticFrame := generator.GenerateStaticFrame()
			segmenter.SegmentMotion(staticFrame)
			staticFrame.Close()
		}

		// Process the motion frame.
		segmenter.SegmentMotion(frame)
		checksums[i] = images.ComputeMatChecksum(segmenter.Threshold)

		segmenter.Kernel.Close()
		segmenter.Close()
	}

	// Verify all runs produced identical results.
	for i := 1; i < len(checksums); i++ {
		if checksums[i] != checksums[0] {
			result.Success = false
			result.Error = fmt.Sprintf("Idempotency violation: checksum[0]=%s, checksum[%d]=%s",
				checksums[0], i, checksums[i])
			t.Error(result.Error)
		}
	}

	result.OutputChecksum = checksums[0]
}

// BenchmarkSubtractBackground measures performance of background subtraction.
func BenchmarkSubtractBackground(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080) // HD resolution.

	segmenter := images.NewMotionSegmenter()
	defer segmenter.Close()

	frame := generator.GenerateStaticFrame()
	defer frame.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		segmenter.SubtractBackground(frame)
	}

	result := &TestResult{
		TestName:  "BenchmarkSubtractBackground",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations": b.N,
			"ns_per_op":  b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution": "1920x1080",
		},
	}
	store.Save(result)
}

// BenchmarkFullPipeline measures end-to-end pipeline performance.
func BenchmarkFullPipeline(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080)

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	frame := generator.GenerateMotionFrame(500, 500, 200)
	defer frame.Close()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = segmenter.SegmentMotion(frame)
	}

	result := &TestResult{
		TestName:  "BenchmarkFullPipeline",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":    b.N,
			"ns_per_op":     b.Elapsed().Nanoseconds() / int64(b.N),
			"fps_potential": float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"resolution":    "1920x1080",
		},
	}
	store.Save(result)
}

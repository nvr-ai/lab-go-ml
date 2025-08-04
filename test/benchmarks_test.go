package test

import (
	"image"
	"runtime"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/motion"
	"gocv.io/x/gocv"
)

// getBenchmarkMemStats returns current memory statistics for benchmark reporting.
func getBenchmarkMemStats() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return map[string]interface{}{
		"heap_alloc_mb":     float64(m.HeapAlloc) / 1024 / 1024,
		"heap_sys_mb":       float64(m.HeapSys) / 1024 / 1024,
		"total_alloc_mb":    float64(m.TotalAlloc) / 1024 / 1024,
		"mallocs":           m.Mallocs,
		"frees":             m.Frees,
		"gc_cycles":         m.NumGC,
		"goroutines":        runtime.NumGoroutine(),
	}
}

// BenchmarkMotionDetectorProcess benchmarks the core motion detection processing logic.
func BenchmarkMotionDetectorProcess(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	
	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 1 * time.Second,
	}
	detector := motion.New(config)
	defer detector.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate motion detection with varying conditions
		detected := i%3 == 0 // Motion in 1/3 of frames
		area := float64(50000 + (i%10000)) // Varying area sizes
		detector.Process(detected, area)
	}

	memStats := getBenchmarkMemStats()
	result := &TestResult{
		TestName:  "BenchmarkMotionDetectorProcess",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"test_type":          "motion_detection_logic",
			"heap_alloc_mb":      memStats["heap_alloc_mb"],
			"goroutines":         memStats["goroutines"],
		},
	}
	store.Save(result)
}

// BenchmarkMotionDetectorFPS benchmarks FPS calculation performance.
func BenchmarkMotionDetectorFPS(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	
	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 1 * time.Second,
	}
	detector := motion.New(config)
	defer detector.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		hasMotion := i%4 == 0 // Motion in 1/4 of frames
		detector.FPS(hasMotion)
	}

	result := &TestResult{
		TestName:  "BenchmarkMotionDetectorFPS",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"test_type":          "fps_calculation",
		},
	}
	store.Save(result)
}

// BenchmarkMotionDetectorMetrics benchmarks metrics collection performance.
func BenchmarkMotionDetectorMetrics(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	
	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 1 * time.Second,
	}
	detector := motion.New(config)
	defer detector.Close()

	// Simulate some activity first
	for i := 0; i < 100; i++ {
		detector.FPS(i%5 == 0)
		detector.Process(i%3 == 0, float64(30000+i*100))
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = detector.CollectMetrics()
	}

	result := &TestResult{
		TestName:  "BenchmarkMotionDetectorMetrics",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"test_type":          "metrics_collection",
		},
	}
	store.Save(result)
}

// BenchmarkApplyThreshold benchmarks the thresholding operation.
func BenchmarkApplyThreshold(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080)

	segmenter := images.NewMotionSegmenter()
	defer segmenter.Close()

	// Setup with background subtraction first
	frame := generator.GenerateStaticFrame()
	segmenter.SubtractBackground(frame)
	frame.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		segmenter.ApplyThreshold(25, 255)
	}

	result := &TestResult{
		TestName:  "BenchmarkApplyThreshold",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1920x1080",
			"test_type":          "thresholding",
		},
	}
	store.Save(result)
}

// BenchmarkFillGaps benchmarks morphological dilation operations.
func BenchmarkFillGaps(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080)

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	// Setup with background subtraction and thresholding first
	frame := generator.GenerateMotionFrame(500, 500, 200)
	segmenter.SubtractBackground(frame)
	segmenter.ApplyThreshold(25, 255)
	frame.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		segmenter.FillGaps()
	}

	result := &TestResult{
		TestName:  "BenchmarkFillGaps",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1920x1080",
			"test_type":          "morphological_operations",
		},
	}
	store.Save(result)
}

// BenchmarkDetectContours benchmarks contour detection performance.
func BenchmarkDetectContours(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080)

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	// Setup with full pipeline preparation
	frame := generator.GenerateMotionFrame(500, 500, 200)
	segmenter.SubtractBackground(frame)
	segmenter.ApplyThreshold(25, 255)
	segmenter.FillGaps()
	frame.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		contours := segmenter.DetectContours()
		contours.Close() // Important: release contours memory
	}

	result := &TestResult{
		TestName:  "BenchmarkDetectContours",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1920x1080",
			"test_type":          "contour_detection",
		},
	}
	store.Save(result)
}

// BenchmarkIntegratedMotionDetection benchmarks the complete end-to-end pipeline.
func BenchmarkIntegratedMotionDetection(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1280, 720) // Standard HD resolution

	// Initialize motion detector
	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 500 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	// Initialize motion segmenter
	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Generate frame with motion in some frames
		var frame gocv.Mat
		if i%5 == 0 {
			frame = generator.GenerateMotionFrame(200+i%400, 200+i%300, 100)
		} else {
			frame = generator.GenerateStaticFrame()
		}

		// Run full segmentation pipeline
		contours := segmenter.SegmentMotion(frame)
		
		// Calculate total area of motion
		totalArea := 0.0
		for j := 0; j < contours.Size(); j++ {
			area := gocv.ContourArea(contours.At(j))
			totalArea += area
		}
		
		// Run motion detection logic
		hasMotion := totalArea > detector.MinimumArea
		detector.FPS(hasMotion)
		detector.Process(hasMotion, totalArea)

		// Cleanup
		contours.Close()
		frame.Close()
	}

	result := &TestResult{
		TestName:  "BenchmarkIntegratedMotionDetection",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1280x720",
			"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":          "end_to_end_pipeline",
		},
	}
	store.Save(result)
}

// BenchmarkMultiResolution benchmarks performance across different resolutions.
func BenchmarkMultiResolution(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	
	resolutions := []struct {
		name   string
		width  int
		height int
	}{
		{"480p", 640, 480},
		{"720p", 1280, 720},
		{"1080p", 1920, 1080},
		{"4K", 3840, 2160},
	}

	for _, res := range resolutions {
		b.Run(res.name, func(b *testing.B) {
			generator := NewMockFrameGenerator(res.width, res.height)
			segmenter := images.NewMotionSegmenter()
			segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
			defer segmenter.Kernel.Close()
			defer segmenter.Close()

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				frame := generator.GenerateMotionFrame(res.width/4, res.height/4, 100)
				contours := segmenter.SegmentMotion(frame)
				contours.Close()
				frame.Close()
			}

			result := &TestResult{
				TestName:  "BenchmarkMultiResolution_" + res.name,
				Timestamp: time.Now(),
				Duration:  time.Duration(b.Elapsed()),
				Success:   true,
				Metadata: map[string]interface{}{
					"operations":         b.N,
					"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
					"resolution":         res.name,
					"width":              res.width,
					"height":             res.height,
					"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
					"test_type":          "resolution_scaling",
				},
			}
			store.Save(result)
		})
	}
}

// BenchmarkMemoryAllocation benchmarks memory allocation patterns.
func BenchmarkMemoryAllocation(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080)

	b.ResetTimer()
	b.ReportAllocs()

	var startMem, endMem runtime.MemStats
	runtime.ReadMemStats(&startMem)

	for i := 0; i < b.N; i++ {
		// Create and destroy segmenter to test allocation patterns
		segmenter := images.NewMotionSegmenter()
		segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
		
		frame := generator.GenerateMotionFrame(500, 500, 200)
		contours := segmenter.SegmentMotion(frame)
		
		// Cleanup
		contours.Close()
		frame.Close()
		segmenter.Kernel.Close()
		segmenter.Close()
	}

	runtime.ReadMemStats(&endMem)

	result := &TestResult{
		TestName:  "BenchmarkMemoryAllocation",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":          b.N,
			"ns_per_op":           b.Elapsed().Nanoseconds() / int64(b.N),
			"heap_start_mb":       float64(startMem.HeapAlloc) / 1024 / 1024,
			"heap_end_mb":         float64(endMem.HeapAlloc) / 1024 / 1024,
			"heap_delta_mb":       float64(endMem.HeapAlloc-startMem.HeapAlloc) / 1024 / 1024,
			"total_alloc_mb":      float64(endMem.TotalAlloc-startMem.TotalAlloc) / 1024 / 1024,
			"gc_cycles":           endMem.NumGC - startMem.NumGC,
			"test_type":           "memory_allocation",
		},
	}
	store.Save(result)
}

// BenchmarkConcurrentProcessing benchmarks concurrent motion detection scenarios.
func BenchmarkConcurrentProcessing(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	
	b.ResetTimer()
	b.ReportAllocs()

	// Test concurrent processing with multiple goroutines
	b.RunParallel(func(pb *testing.PB) {
		generator := NewMockFrameGenerator(1280, 720)
		segmenter := images.NewMotionSegmenter()
		segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
		defer segmenter.Kernel.Close()
		defer segmenter.Close()

		config := motion.Config{
			MinimumArea:       30000,
			MinMotionDuration: 500 * time.Millisecond,
		}
		detector := motion.New(config)
		defer detector.Close()

		i := 0
		for pb.Next() {
			frame := generator.GenerateMotionFrame(200+i%400, 200+i%300, 100)
			contours := segmenter.SegmentMotion(frame)
			
			totalArea := 0.0
			for j := 0; j < contours.Size(); j++ {
				area := gocv.ContourArea(contours.At(j))
				totalArea += area
			}
			
			hasMotion := totalArea > detector.MinimumArea
			detector.Process(hasMotion, totalArea)
			
			contours.Close()
			frame.Close()
			i++
		}
	})

	result := &TestResult{
		TestName:  "BenchmarkConcurrentProcessing",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"goroutines":         runtime.NumGoroutine(),
			"cpu_count":          runtime.NumCPU(),
			"test_type":          "concurrent_processing",
		},
	}
	store.Save(result)
}
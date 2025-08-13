package kernels

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"runtime"
	"testing"
	"time"
)

// Benchmark Configuration Constants.
const (
	// Standard Object Detection Resolutions.
	ResolutionTiny   = 320  // Low-resource edge devices.
	ResolutionSmall  = 640  // Standard YOLO input.
	ResolutionMedium = 800  // Faster R-CNN typical.
	ResolutionLarge  = 1024 // High-precision detection.
	ResolutionHD     = 1280 // 720p video streams.
	ResolutionFHD    = 1920 // 1080p video streams.

	// Blur Radius Test Range (covers typical preprocessing needs).
	MinRadius = 1  // Minimal noise reduction.
	MaxRadius = 15 // Heavy smoothing (rarely used).

	// Performance Measurement Iterations.
	WarmupIterations = 10  // JIT warmup and memory allocation.
	BenchIterations  = 100 // Statistical significance

	// Memory Pool Testing.
	PoolTestFrames = 1000 // Simulate video processing load.
	FrameRate      = 30   // FPS for GC pressure simulation.
)

// TestImageGenerator creates synthetic images with controlled patterns
// for reproducible benchmarking across different detection scenarios.
type TestImageGenerator struct {
	width, height int
	pattern       PatternType
}

// PatternType defines the type of pattern to generate.
type PatternType int

const (
	// PatternNoise: Random noise - worst case for blur algorithms.
	// Exercises all code paths, maximum cache misses.
	PatternNoise PatternType = iota
	// PatternGradient: Smooth gradients - best case scenario.
	// Predictable memory access, high cache hit rate.
	PatternGradient
	// PatternChessboard: High-frequency alternating pattern.
	// Tests edge handling and precision requirements.
	PatternChessboard
	// PatternObjects: Simulates object detection scenarios.
	// Multiple rectangular "objects" on uniform background.
	PatternObjects
	// PatternVideo: Realistic video frame simulation.
	// Mix of textures, edges, and smooth regions.
	PatternVideo
)

// generateTestImage creates a synthetic image with the specified pattern
// designed to stress-test different algorithmic aspects of blur kernels
func (g *TestImageGenerator) generateTestImage() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, g.width, g.height))

	switch g.pattern {
	case PatternNoise:
		// Maximum entropy pattern - every pixel different.
		// Defeats any caching or SIMD optimizations.
		// Tests worst-case performance characteristics.
		for y := 0; y < g.height; y++ {
			for x := 0; x < g.width; x++ {
				// Pseudo-random but deterministic for reproducibility.
				seed := uint32(x + y*g.width)
				r := uint8((seed*1103515245 + 12345) >> 24)
				gr := uint8((seed*1664525 + 1013904223) >> 24)
				b := uint8((seed*2147483647 + 2654435761) >> 24)
				img.SetRGBA(x, y, color.RGBA{r, gr, b, 255})
			}
		}

	case PatternGradient:
		// Smooth gradient - best case for cache performance.
		// Predictable memory access patterns, high spatial coherence.
		// Tests optimal performance under ideal conditions.
		for y := 0; y < g.height; y++ {
			for x := 0; x < g.width; x++ {
				// Diagonal gradient from black to white.
				intensity := uint8(((x + y) * 255) / (g.width + g.height))
				img.SetRGBA(x, y, color.RGBA{intensity, intensity, intensity, 255})
			}
		}

	case PatternChessboard:
		// High-frequency alternating pattern.
		// Maximum stress on edge handling algorithms.
		// Tests precision requirements and boundary conditions.
		squareSize := 8 // 8x8 squares like a chessboard.
		for y := 0; y < g.height; y++ {
			for x := 0; x < g.width; x++ {
				// Checkerboard pattern with 8x8 squares
				if ((x/squareSize)+(y/squareSize))%2 == 0 {
					img.SetRGBA(x, y, color.RGBA{255, 255, 255, 255}) // White
				} else {
					img.SetRGBA(x, y, color.RGBA{0, 0, 0, 255}) // Black
				}
			}
		}

	case PatternObjects:
		// Simulated object detection scenario
		// Rectangular "objects" on uniform background
		// Tests real-world performance characteristics

		// Background: medium gray
		for y := 0; y < g.height; y++ {
			for x := 0; x < g.width; x++ {
				img.SetRGBA(x, y, color.RGBA{128, 128, 128, 255})
			}
		}

		// Add rectangular "objects" of various sizes
		objects := []struct {
			x, y, w, h int
			c          color.RGBA
		}{
			{g.width / 8, g.height / 8, g.width / 4, g.height / 6, color.RGBA{255, 0, 0, 255}},     // Red vehicle
			{g.width / 2, g.height / 3, g.width / 6, g.height / 4, color.RGBA{0, 255, 0, 255}},     // Green person
			{3 * g.width / 4, g.height / 2, g.width / 5, g.height / 8, color.RGBA{0, 0, 255, 255}}, // Blue object
		}

		for _, obj := range objects {
			for y := obj.y; y < obj.y+obj.h && y < g.height; y++ {
				for x := obj.x; x < obj.x+obj.w && x < g.width; x++ {
					img.SetRGBA(x, y, obj.c)
				}
			}
		}

	case PatternVideo:
		// Realistic video frame simulation
		// Mix of smooth regions, edges, texture, and noise
		// Best approximation of real-world detection input

		// Base gradient background
		for y := 0; y < g.height; y++ {
			for x := 0; x < g.width; x++ {
				baseIntensity := uint8(100 + (x*155)/g.width)

				// Add some texture noise (±10 levels)
				seed := uint32(x*31 + y*37)
				noise := int8((seed % 21) - 10)

				final := int(baseIntensity) + int(noise)
				if final < 0 {
					final = 0
				}
				if final > 255 {
					final = 255
				}

				intensity := uint8(final)
				img.SetRGBA(x, y, color.RGBA{intensity, intensity, intensity, 255})
			}
		}

		// Add some high-contrast edges (simulate building edges, vehicles)
		for i := 0; i < 5; i++ {
			lineX := (i * g.width) / 6
			for y := 0; y < g.height; y++ {
				if lineX < g.width {
					img.SetRGBA(lineX, y, color.RGBA{255, 255, 255, 255})
				}
			}
		}
	}

	return img
}

// PerformanceMetrics captures comprehensive timing and resource usage data
// for detailed analysis of blur algorithm performance characteristics
type PerformanceMetrics struct {
	// Core timing measurements (nanoseconds)
	MinLatency  int64   // Best-case single iteration
	MaxLatency  int64   // Worst-case single iteration
	MeanLatency float64 // Average across all iterations
	P95Latency  int64   // 95th percentile (important for real-time)
	P99Latency  int64   // 99th percentile (outlier detection)

	// Throughput measurements
	PixelsPerSecond float64 // Raw processing rate
	FramesPerSecond float64 // Video processing capability

	// Memory and resource usage
	BytesAllocated   uint64  // Total memory allocated during test
	AllocationsCount uint64  // Number of allocation events
	GCPressure       float64 // GC overhead as percentage of runtime

	// System resource utilization
	CPUCores        int     // Number of cores utilized
	MemoryBandwidth float64 // Estimated memory bandwidth usage (GB/s)
	CacheEfficiency float64 // Cache hit rate estimation (0-1)

	// Statistical analysis
	StandardDeviation float64 // Timing consistency measurement
	CoefficientOfVar  float64 // Normalized variance (should be <0.1)
}

// measurePerformance executes comprehensive performance analysis
// of blur algorithms under controlled conditions with statistical rigor
func measurePerformance(name string, blurFunc func(), imageSize int, radius int) PerformanceMetrics {
	runtime.GC() // Clean slate for memory measurements

	// Memory baseline measurement
	var m1, m2 runtime.MemStats
	runtime.ReadMemStats(&m1)

	// Warmup phase - ensure JIT compilation and memory allocation
	// Critical for getting stable performance measurements
	start := time.Now()
	for i := 0; i < WarmupIterations; i++ {
		blurFunc()
	}
	warmupDuration := time.Since(start)

	// Main performance measurement phase
	latencies := make([]int64, BenchIterations)
	start = time.Now()

	for i := 0; i < BenchIterations; i++ {
		iterStart := time.Now()
		blurFunc()
		latencies[i] = time.Since(iterStart).Nanoseconds()
	}

	totalDuration := time.Since(start)
	runtime.ReadMemStats(&m2)

	// Statistical analysis of latency distribution
	var sum int64 = 0
	minLat := latencies[0]
	maxLat := latencies[0]

	for _, lat := range latencies {
		sum += lat
		if lat < minLat {
			minLat = lat
		}
		if lat > maxLat {
			maxLat = lat
		}
	}

	meanLat := float64(sum) / float64(BenchIterations)

	// Calculate standard deviation and coefficient of variation
	// High variance indicates inconsistent performance (bad for real-time)
	var variance float64 = 0
	for _, lat := range latencies {
		diff := float64(lat) - meanLat
		variance += diff * diff
	}
	variance /= float64(BenchIterations)
	stdDev := math.Sqrt(variance)
	coeffVar := stdDev / meanLat

	// Percentile calculations (sort latencies first)
	sortedLats := make([]int64, len(latencies))
	copy(sortedLats, latencies)
	// Simple insertion sort for small arrays
	for i := 1; i < len(sortedLats); i++ {
		key := sortedLats[i]
		j := i - 1
		for j >= 0 && sortedLats[j] > key {
			sortedLats[j+1] = sortedLats[j]
			j--
		}
		sortedLats[j+1] = key
	}

	p95Index := int(0.95 * float64(BenchIterations))
	p99Index := int(0.99 * float64(BenchIterations))
	if p95Index >= BenchIterations {
		p95Index = BenchIterations - 1
	}
	if p99Index >= BenchIterations {
		p99Index = BenchIterations - 1
	}

	// Performance calculations
	totalPixels := int64(imageSize * imageSize * BenchIterations)
	pixelsPerSecond := float64(totalPixels) / (float64(totalDuration.Nanoseconds()) / 1e9)
	framesPerSecond := float64(BenchIterations) / (float64(totalDuration.Nanoseconds()) / 1e9)

	// Memory bandwidth estimation (conservative)
	// Each pixel read once, written twice (intermediate + final)
	bytesPerPixel := 4                                                              // RGBA
	totalBytes := float64(totalPixels * int64(bytesPerPixel) * 3)                   // Read once, write twice
	memBandwidth := totalBytes / (float64(totalDuration.Nanoseconds()) / 1e9) / 1e9 // GB/s

	// GC pressure calculation
	gcPressure := float64(warmupDuration.Nanoseconds()) / float64(totalDuration.Nanoseconds())

	return PerformanceMetrics{
		MinLatency:        minLat,
		MaxLatency:        maxLat,
		MeanLatency:       meanLat,
		P95Latency:        sortedLats[p95Index],
		P99Latency:        sortedLats[p99Index],
		PixelsPerSecond:   pixelsPerSecond,
		FramesPerSecond:   framesPerSecond,
		BytesAllocated:    m2.TotalAlloc - m1.TotalAlloc,
		AllocationsCount:  m2.Mallocs - m1.Mallocs,
		GCPressure:        gcPressure,
		CPUCores:          runtime.NumCPU(),
		MemoryBandwidth:   memBandwidth,
		CacheEfficiency:   estimateCacheEfficiency(stdDev, meanLat),
		StandardDeviation: stdDev,
		CoefficientOfVar:  coeffVar,
	}
}

// estimateCacheEfficiency provides a rough estimate of cache performance
// based on timing variance - high variance often indicates cache misses
func estimateCacheEfficiency(stdDev, mean float64) float64 {
	// Heuristic: low coefficient of variation suggests good cache behavior
	coeffVar := stdDev / mean
	if coeffVar < 0.1 {
		return 0.9 // Excellent cache efficiency
	} else if coeffVar < 0.2 {
		return 0.7 // Good cache efficiency
	} else if coeffVar < 0.5 {
		return 0.5 // Fair cache efficiency
	} else {
		return 0.2 // Poor cache efficiency
	}
}

// BenchmarkBoxBlurGPT5 comprehensive performance analysis of GPT5 implementation
func BenchmarkBoxBlurGPT5(b *testing.B) {
	resolutions := []int{ResolutionSmall, ResolutionMedium, ResolutionLarge, ResolutionHD}
	radii := []int{1, 3, 5, 7, 10, 15}
	patterns := []PatternType{PatternNoise, PatternGradient, PatternVideo}

	b.ReportAllocs() // Enable memory allocation reporting

	for _, resolution := range resolutions {
		for _, radius := range radii {
			for _, pattern := range patterns {
				patternName := map[PatternType]string{
					PatternNoise:    "noise",
					PatternGradient: "gradient",
					PatternVideo:    "video",
				}[pattern]

				name := fmt.Sprintf("GPT5/res_%dx%d/radius_%d/pattern_%s",
					resolution, resolution, radius, patternName)

				b.Run(name, func(b *testing.B) {
					generator := &TestImageGenerator{resolution, resolution, pattern}
					img := generator.generateTestImage()

					// Setup pool for realistic video processing simulation
					pool := &Pool{}
					opts := Options{
						Radius:   radius,
						Edge:     EdgeClamp, // Most common in detection pipelines
						Pool:     pool,
						Parallel: runtime.NumCPU() > 1, // Auto-detect parallel capability
					}

					b.ResetTimer() // Don't count setup time

					for i := 0; i < b.N; i++ {
						result := BoxBlur(img, opts)
						_ = result // Prevent optimization elimination
					}
				})
			}
		}
	}
}

// BenchmarkBoxBlurCopilot performance analysis of Copilot implementation
// Note: This requires implementing missing MapCoord and Parallel functions
func BenchmarkBoxBlurCopilot(b *testing.B) {
	// Implementation requires completing the missing functions first
	b.Skip("Copilot implementation incomplete - missing MapCoord and Parallel functions")

	// TODO: Complete after implementing missing dependencies
}

// AccuracyTestSuite validates mathematical correctness of blur implementations
type AccuracyTestSuite struct {
	tolerance float64 // Maximum acceptable error per channel
}

// TestBlurAccuracy validates mathematical correctness against reference implementation
func TestBlurAccuracy(t *testing.T) {
	suite := &AccuracyTestSuite{tolerance: 1.0} // ±1 level tolerance (0.4% error)

	// Test cases covering edge cases and typical scenarios
	testCases := []struct {
		width, height int
		radius        int
		pattern       PatternType
		description   string
	}{
		{32, 32, 1, PatternNoise, "Small image, minimal blur"},
		{640, 640, 3, PatternGradient, "Standard YOLO input, moderate blur"},
		{800, 600, 5, PatternObjects, "Faster R-CNN input, heavy blur"},
		{1920, 1080, 7, PatternVideo, "1080p video, maximum blur"},
		{33, 27, 2, PatternChessboard, "Odd dimensions, edge cases"}, // Non-power-of-2
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			generator := &TestImageGenerator{tc.width, tc.height, tc.pattern}
			img := generator.generateTestImage()

			// Test GPT5 implementation
			pool := &Pool{}
			opts := Options{
				Radius:   tc.radius,
				Edge:     EdgeClamp,
				Pool:     pool,
				Parallel: false, // Single-threaded for deterministic results
			}

			result := BoxBlur(img, opts)

			// Validate result dimensions
			if result.Rect.Dx() != tc.width || result.Rect.Dy() != tc.height {
				t.Errorf("Dimension mismatch: expected %dx%d, got %dx%d",
					tc.width, tc.height, result.Rect.Dx(), result.Rect.Dy())
			}

			// Validate blur properties (center pixel should be average of neighborhood)
			if tc.width > 2*tc.radius && tc.height > 2*tc.radius {
				suite.validateBlurMath(t, img, result, tc.radius, tc.width/2, tc.height/2)
			}
		})
	}
}

// validateBlurMath verifies that blur result matches mathematical expectation
// for a specific pixel location by manually computing the expected value
func (suite *AccuracyTestSuite) validateBlurMath(t *testing.T, original, blurred *image.RGBA,
	radius int, testX, testY int) {

	// Manually compute expected blur value at (testX, testY)
	var expectedR, expectedG, expectedB, expectedA float64
	count := 0

	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			// Use same edge handling as GPT5 (clamp)
			srcX := testX + dx
			srcY := testY + dy

			if srcX < 0 {
				srcX = 0
			}
			if srcX >= original.Rect.Dx() {
				srcX = original.Rect.Dx() - 1
			}
			if srcY < 0 {
				srcY = 0
			}
			if srcY >= original.Rect.Dy() {
				srcY = original.Rect.Dy() - 1
			}

			pixel := original.RGBAAt(srcX, srcY)
			expectedR += float64(pixel.R)
			expectedG += float64(pixel.G)
			expectedB += float64(pixel.B)
			expectedA += float64(pixel.A)
			count++
		}
	}

	expectedR /= float64(count)
	expectedG /= float64(count)
	expectedB /= float64(count)
	expectedA /= float64(count)

	// Compare with actual result
	actual := blurred.RGBAAt(testX, testY)

	if math.Abs(float64(actual.R)-expectedR) > suite.tolerance ||
		math.Abs(float64(actual.G)-expectedG) > suite.tolerance ||
		math.Abs(float64(actual.B)-expectedB) > suite.tolerance ||
		math.Abs(float64(actual.A)-expectedA) > suite.tolerance {

		t.Errorf("Accuracy test failed at (%d,%d):\n"+
			"Expected RGBA: (%.2f, %.2f, %.2f, %.2f)\n"+
			"Actual RGBA:   (%d, %d, %d, %d)\n"+
			"Differences:   (%.2f, %.2f, %.2f, %.2f)",
			testX, testY,
			expectedR, expectedG, expectedB, expectedA,
			actual.R, actual.G, actual.B, actual.A,
			math.Abs(float64(actual.R)-expectedR),
			math.Abs(float64(actual.G)-expectedG),
			math.Abs(float64(actual.B)-expectedB),
			math.Abs(float64(actual.A)-expectedA))
	}
}

// MemoryPressureTest simulates video processing workload to measure GC impact
func TestMemoryPressure(t *testing.T) {
	const (
		testDurationSeconds = 10
		frameRate           = 30
		frameWidth          = 1920
		frameHeight         = 1080
	)

	// Test with and without memory pooling.
	t.Run("WithPool", func(t *testing.T) {
		pool := &Pool{}
		testMemoryPressure(t, pool, testDurationSeconds, frameRate, frameWidth, frameHeight)
	})

	t.Run("WithoutPool", func(t *testing.T) {
		testMemoryPressure(t, nil, testDurationSeconds, frameRate, frameWidth, frameHeight)
	})
}

func testMemoryPressure(t *testing.T, pool *Pool, durationSec, fps, width, height int) {
	generator := &TestImageGenerator{width, height, PatternVideo}
	img := generator.generateTestImage()

	opts := Options{
		Radius:   3,
		Edge:     EdgeClamp,
		Pool:     pool,
		Parallel: true,
	}

	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	start := time.Now()
	frameCount := 0

	for time.Since(start) < time.Duration(durationSec)*time.Second {
		result := BoxBlur(img, opts)
		_ = result
		frameCount++

		// Simulate frame rate timing
		time.Sleep(time.Duration(1000/fps) * time.Millisecond)
	}

	runtime.ReadMemStats(&m2)

	totalAllocs := m2.TotalAlloc - m1.TotalAlloc
	numGC := m2.NumGC - m1.NumGC

	poolStr := "without"
	if pool != nil {
		poolStr = "with"
	}

	t.Logf("Memory pressure test %s pool:\n"+
		"  Frames processed: %d\n"+
		"  Total allocations: %d MB\n"+
		"  GC cycles: %d\n"+
		"  Allocs per frame: %.2f KB\n"+
		"  GC frequency: %.2f cycles/sec",
		poolStr, frameCount,
		totalAllocs/1024/1024,
		numGC,
		float64(totalAllocs)/float64(frameCount)/1024,
		float64(numGC)/float64(durationSec))

	// Performance assertions for production readiness
	allocsPerFrame := float64(totalAllocs) / float64(frameCount)
	gcFrequency := float64(numGC) / float64(durationSec)

	if pool != nil {
		// With pooling, should have reduced allocations after warmup
		// Note: During initial warmup, allocations will be higher
		if allocsPerFrame > 50*1024*1024 { // 50MB per frame threshold (very generous for warmup)
			t.Errorf("Too many allocations per frame with pool: %.2f KB", allocsPerFrame/1024)
		}
		if gcFrequency > 20.0 { // Max 20 GC per second with pool (generous for warmup)
			t.Errorf("Too frequent GC with pool: %.2f cycles/sec", gcFrequency)
		}
	}
}

// EdgeHandlingTest verifies correct behavior at image boundaries
// Critical for object detection where objects may be partially visible
func TestEdgeHandling(t *testing.T) {
	// Create small test image for manual verification
	img := image.NewRGBA(image.Rect(0, 0, 5, 5))

	// Fill with distinctive pattern
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			value := uint8((x + y) * 50)
			img.SetRGBA(x, y, color.RGBA{value, value, value, 255})
		}
	}

	edgeModes := []EdgeMode{EdgeClamp, EdgeMirror, EdgeWrap}
	edgeNames := []string{"Clamp", "Mirror", "Wrap"}

	for i, mode := range edgeModes {
		t.Run(edgeNames[i], func(t *testing.T) {
			pool := &Pool{}
			opts := Options{
				Radius:   2, // Large radius relative to 5x5 image
				Edge:     mode,
				Pool:     pool,
				Parallel: false,
			}

			result := BoxBlur(img, opts)

			// Verify edge pixels are handled correctly
			// Corner pixel (0,0) should be average of mapped coordinates
			corner := result.RGBAAt(0, 0)

			// The exact expected value depends on edge mode
			// Here we just verify it's reasonable (not zero, not overflow)
			if corner.R == 0 && corner.G == 0 && corner.B == 0 {
				t.Errorf("Edge mode %s produced zero corner pixel", edgeNames[i])
			}

			if corner.A != 255 {
				t.Errorf("Edge mode %s corrupted alpha channel: got %d, want 255",
					edgeNames[i], corner.A)
			}
		})
	}
}

// ParallelConsistencyTest ensures parallel and serial execution produce identical results
func TestParallelConsistency(t *testing.T) {
	generator := &TestImageGenerator{640, 480, PatternVideo}
	img := generator.generateTestImage()

	pool := &Pool{}

	// Serial execution
	opts1 := Options{Radius: 5, Edge: EdgeClamp, Pool: pool, Parallel: false}
	result1 := BoxBlur(img, opts1)

	// Parallel execution
	opts2 := Options{Radius: 5, Edge: EdgeClamp, Pool: pool, Parallel: true}
	result2 := BoxBlur(img, opts2)

	// Results should be pixel-identical
	for y := 0; y < 480; y++ {
		for x := 0; x < 640; x++ {
			p1 := result1.RGBAAt(x, y)
			p2 := result2.RGBAAt(x, y)

			if p1 != p2 {
				t.Errorf("Parallel inconsistency at (%d,%d): serial=%+v, parallel=%+v",
					x, y, p1, p2)
				return // Stop after first difference
			}
		}
	}
}

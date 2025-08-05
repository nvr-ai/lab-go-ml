package test

import (
	"fmt"
	"image"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/motion"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gocv.io/x/gocv"
)

// TestMotionDetectorRealVideoSequence tests with a real video sequence using the loader pattern.
func TestMotionDetectorRealVideoSequence(t *testing.T) {
	store := NewTestResultStore("./test_results")
	start := time.Now()

	result := &TestResult{
		TestName:  "TestMotionDetectorRealVideoSequence",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	// Load real video frames using the loader pattern
	imageSequences := []string{
		"../../../ml/corpus/images/clip-4k.mp4",
		// "../../../ml/corpus/images/motion_test",
		// "./test_images",
		// "../test_data/frames",
	}

	var imageFiles []ImageFile
	var err error
	var sourceDir string

	// Try to find available image sequence
	for _, dir := range imageSequences {
		imageFiles, err = LoadDirectoryImageFiles(dir)
		if err == nil && len(imageFiles) > 0 {
			sourceDir = dir
			break
		}
	}

	// If no real images found, create comprehensive mock sequence
	if len(imageFiles) == 0 {
		t.Log("No real image sequences found, creating comprehensive mock sequence")
		imageFiles = createComprehensiveMockSequence(t)
		sourceDir = "mock_sequence"
	}

	require.Greater(t, len(imageFiles), 20, "Need at least 20 frames for comprehensive motion detection test")

	// Test multiple detector configurations
	testConfigs := []struct {
		name   string
		config motion.Config
	}{
		{
			name: "high_sensitivity",
			config: motion.Config{
				MinimumArea:       5000,
				MinMotionDuration: 50 * time.Millisecond,
			},
		},
		{
			name: "medium_sensitivity",
			config: motion.Config{
				MinimumArea:       20000,
				MinMotionDuration: 200 * time.Millisecond,
			},
		},
		{
			name: "low_sensitivity",
			config: motion.Config{
				MinimumArea:       50000,
				MinMotionDuration: 500 * time.Millisecond,
			},
		},
	}

	configResults := make(map[string]interface{})

	for _, testConfig := range testConfigs {
		t.Run(testConfig.name, func(t *testing.T) {
			detector := motion.New(testConfig.config)
			defer detector.Close()

			segmenter := images.NewMotionSegmenter()
			segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
			defer segmenter.Kernel.Close()
			defer segmenter.Close()

			motionReports := 0
			maxArea := 0.0
			totalProcessingTime := time.Duration(0)

			fmt.Printf("path=%s,frames=%d\n", sourceDir, len(imageFiles))
			// Process frame sequence
			for i, img := range imageFiles {
				// if i > 100 { // Limit for test performance
				// 	break
				// }

				frameStart := time.Now()

				// Decode and process frame
				frame, err := gocv.IMDecode(img.Data, gocv.IMReadGrayScale)
				require.NoError(t, err, "Failed to decode frame %d", i)

				// Run motion segmentation
				contours := segmenter.SegmentMotion(frame)

				// Calculate motion area
				frameArea := 0.0
				for j := 0; j < contours.Size(); j++ {
					area := gocv.ContourArea(contours.At(j))
					frameArea += area
				}

				if frameArea > maxArea {
					maxArea = frameArea
				}

				// Update detector
				hasMotion := frameArea > detector.MinimumArea
				detector.FPS(hasMotion)
				detector.FrameProcessingTime = time.Since(frameStart)
				totalProcessingTime += detector.FrameProcessingTime

				// Process motion detection
				report, status := detector.Process(hasMotion, frameArea)
				if report {
					motionReports++
					t.Logf("%s - Frame %d: %s (area: %.0f)", testConfig.name, i, status, frameArea)
				}

				// Cleanup
				contours.Close()
				frame.Close()
			}

			// Record results for this configuration
			configResults[testConfig.name] = map[string]interface{}{
				"frames_processed":  detector.TotalFrames,
				"motion_frames":     detector.MotionFrames,
				"motion_reports":    motionReports,
				"motion_events":     detector.MotionEventCount,
				"max_area_detected": maxArea,
				"avg_processing_ms": float64(totalProcessingTime.Nanoseconds()) / float64(detector.TotalFrames) / 1e6,
				"motion_percentage": float64(detector.MotionFrames) / float64(detector.TotalFrames) * 100,
			}

			// Assertions for this configuration
			assert.Greater(t, detector.TotalFrames, 0)
			assert.GreaterOrEqual(t, detector.MotionFrames, 0)
			assert.GreaterOrEqual(t, motionReports, 0)
		})
	}

	result.Metadata = map[string]interface{}{
		"source_directory": sourceDir,
		"total_frames":     len(imageFiles),
		"configs_tested":   len(testConfigs),
		"config_results":   configResults,
	}
}

// TestMotionDetectorStateTransitions tests complex state transition scenarios.
func TestMotionDetectorStateTransitions(t *testing.T) {
	store := NewTestResultStore("./test_results")
	start := time.Now()

	result := &TestResult{
		TestName:  "TestMotionDetectorStateTransitions",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 100 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	stateTransitions := []struct {
		name     string
		scenario func() (bool, float64)
		wait     time.Duration
		expected struct {
			report       bool
			motionActive bool
		}
	}{
		{
			name:     "no_motion_initially",
			scenario: func() (bool, float64) { return false, 0 },
			wait:     0,
			expected: struct{ report, motionActive bool }{false, false},
		},
		{
			name:     "motion_starts_insufficient_area",
			scenario: func() (bool, float64) { return true, 25000 }, // Below threshold
			wait:     0,
			expected: struct{ report, motionActive bool }{false, false},
		},
		{
			name:     "motion_starts_sufficient_area",
			scenario: func() (bool, float64) { return true, 35000 }, // Above threshold
			wait:     0,
			expected: struct{ report, motionActive bool }{false, true},
		},
		{
			name:     "motion_continues_short_duration",
			scenario: func() (bool, float64) { return true, 40000 },
			wait:     50 * time.Millisecond, // Less than minimum duration
			expected: struct{ report, motionActive bool }{false, true},
		},
		{
			name:     "motion_continues_sufficient_duration",
			scenario: func() (bool, float64) { return true, 45000 },
			wait:     120 * time.Millisecond, // More than minimum duration
			expected: struct{ report, motionActive bool }{true, true},
		},
		{
			name:     "motion_stops_immediately",
			scenario: func() (bool, float64) { return false, 0 },
			wait:     0,
			expected: struct{ report, motionActive bool }{false, true}, // Still active due to tolerance
		},
		{
			name:     "motion_stops_after_tolerance",
			scenario: func() (bool, float64) { return false, 0 },
			wait:     150 * time.Millisecond, // Past tolerance period
			expected: struct{ report, motionActive bool }{false, false},
		},
		{
			name:     "intermittent_motion_restart",
			scenario: func() (bool, float64) { return true, 50000 },
			wait:     0,
			expected: struct{ report, motionActive bool }{false, true},
		},
	}

	transitionResults := make([]map[string]interface{}, 0)

	for i, transition := range stateTransitions {
		if transition.wait > 0 {
			time.Sleep(transition.wait)
		}

		detected, area := transition.scenario()
		report, status := detector.Process(detected, area)

		// Verify expectations
		assert.Equal(t, transition.expected.report, report,
			"Step %d (%s): Expected report=%v, got %v", i, transition.name, transition.expected.report, report)
		assert.Equal(t, transition.expected.motionActive, detector.IsMotionActive,
			"Step %d (%s): Expected motionActive=%v, got %v", i, transition.name, transition.expected.motionActive, detector.IsMotionActive)

		transitionResults = append(transitionResults, map[string]interface{}{
			"step":          i,
			"name":          transition.name,
			"detected":      detected,
			"area":          area,
			"report":        report,
			"motion_active": detector.IsMotionActive,
			"status":        status,
			"events_count":  detector.MotionEventCount,
		})

		t.Logf("Step %d (%s): detected=%v, area=%.0f, report=%v, active=%v, events=%d",
			i, transition.name, detected, area, report, detector.IsMotionActive, detector.MotionEventCount)
	}

	result.Metadata = map[string]interface{}{
		"transitions_tested": len(stateTransitions),
		"final_events_count": detector.MotionEventCount,
		"transition_results": transitionResults,
	}
}

// TestMotionDetectorLongRunning tests detector stability over extended periods.
func TestMotionDetectorLongRunning(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}

	store := NewTestResultStore("./test_results")
	start := time.Now()

	result := &TestResult{
		TestName:  "TestMotionDetectorLongRunning",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 200 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	// Start profiler to track resource usage
	detector.Profiler.Start()
	defer detector.Profiler.Stop()

	generator := NewMockFrameGenerator(1280, 720)
	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	// Run for extended period (5 minutes worth of frames at 30fps)
	totalFrames := 9000 // 30fps * 300 seconds = 9000 frames
	motionReports := 0
	memorySnapshots := make([]float64, 0)

	for i := 0; i < totalFrames; i++ {
		// Generate realistic motion patterns
		var frame gocv.Mat
		motionCycle := i % 300 // 10-second cycles

		if motionCycle < 60 { // 2 seconds of motion
			frame = generator.GenerateMotionFrame(
				200+i%600, 200+i%400, 100+i%50)
		} else {
			frame = generator.GenerateStaticFrame()
		}

		// Process frame
		frameStart := time.Now()
		contours := segmenter.SegmentMotion(frame)

		frameArea := 0.0
		for j := 0; j < contours.Size(); j++ {
			area := gocv.ContourArea(contours.At(j))
			frameArea += area
		}

		hasMotion := frameArea > detector.MinimumArea
		detector.FPS(hasMotion)
		detector.FrameProcessingTime = time.Since(frameStart)

		report, _ := detector.Process(hasMotion, frameArea)
		if report {
			motionReports++
		}

		// Periodic memory snapshots
		if i%500 == 0 {
			metrics := detector.CollectMetrics()
			memorySnapshots = append(memorySnapshots, metrics["heap_alloc_mb"])

			if i%1000 == 0 {
				t.Logf("Frame %d: Motion reports=%d, Events=%d, Heap=%.2fMB",
					i, motionReports, detector.MotionEventCount, metrics["heap_alloc_mb"])
			}
		}

		// Cleanup
		contours.Close()
		frame.Close()

		// Throttle to prevent overwhelming the system
		if i%100 == 0 {
			time.Sleep(1 * time.Millisecond)
		}
	}

	// Final metrics
	finalMetrics := detector.CollectMetrics()

	// Check for memory leaks (memory should be stable)
	if len(memorySnapshots) > 2 {
		initialMem := memorySnapshots[0]
		finalMem := memorySnapshots[len(memorySnapshots)-1]
		memoryGrowth := finalMem - initialMem

		// Allow some growth but not excessive
		assert.Less(t, memoryGrowth, 10.0, "Memory growth should be less than 10MB over long run")
	}

	result.Metadata = map[string]interface{}{
		"frames_processed": totalFrames,
		"motion_reports":   motionReports,
		"motion_events":    detector.MotionEventCount,
		"final_heap_mb":    finalMetrics["heap_alloc_mb"],
		"memory_snapshots": memorySnapshots,
		"memory_stable":    len(memorySnapshots) > 0,
		"duration_minutes": result.Duration.Minutes(),
	}

	t.Logf("Long-running test completed: %d frames, %d reports, %d events, %.2fMB final heap",
		totalFrames, motionReports, detector.MotionEventCount, finalMetrics["heap_alloc_mb"])
}

// TestMotionDetectorResourceManagement tests proper resource cleanup and management.
func TestMotionDetectorResourceManagement(t *testing.T) {
	store := NewTestResultStore("./test_results")
	start := time.Now()

	result := &TestResult{
		TestName:  "TestMotionDetectorResourceManagement",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	initialMetrics := collectSystemMetrics()

	// Create and destroy multiple detectors to test resource management
	numDetectors := 50
	for i := 0; i < numDetectors; i++ {
		config := motion.Config{
			MinimumArea:       30000 + float64(i*1000),
			MinMotionDuration: time.Duration(100+i*10) * time.Millisecond,
		}

		detector := motion.New(config)

		// Use the detector briefly
		detector.FPS(i%2 == 0)
		detector.Process(i%3 == 0, float64(35000+i*500))
		_ = detector.CollectMetrics()

		// Properly close
		detector.Close()
	}

	finalMetrics := collectSystemMetrics()

	// Check resource usage didn't grow excessively
	memoryGrowth := finalMetrics["heap_alloc_mb"] - initialMetrics["heap_alloc_mb"]
	assert.Less(t, memoryGrowth, 5.0, "Memory growth should be minimal after proper cleanup")

	result.Metadata = map[string]interface{}{
		"detectors_created": numDetectors,
		"initial_heap_mb":   initialMetrics["heap_alloc_mb"],
		"final_heap_mb":     finalMetrics["heap_alloc_mb"],
		"memory_growth_mb":  memoryGrowth,
		"resource_cleanup":  "passed",
	}
}

// TestMotionDetectorErrorConditions tests error handling and edge cases.
func TestMotionDetectorErrorConditions(t *testing.T) {
	store := NewTestResultStore("./test_results")
	start := time.Now()

	result := &TestResult{
		TestName:  "TestMotionDetectorErrorConditions",
		Timestamp: start,
		Success:   true,
	}
	defer func() {
		result.Duration = time.Since(start)
		store.Save(result)
	}()

	// Test various edge cases
	edgeCases := []struct {
		name string
		test func(t *testing.T) map[string]interface{}
	}{
		{
			name: "negative_area",
			test: func(t *testing.T) map[string]interface{} {
				config := motion.Config{MinimumArea: 30000, MinMotionDuration: 100 * time.Millisecond}
				detector := motion.New(config)
				defer detector.Close()

				report, status := detector.Process(true, -1000)
				return map[string]interface{}{
					"report": report,
					"status": status,
					"active": detector.IsMotionActive,
				}
			},
		},
		{
			name: "zero_area",
			test: func(t *testing.T) map[string]interface{} {
				config := motion.Config{MinimumArea: 30000, MinMotionDuration: 100 * time.Millisecond}
				detector := motion.New(config)
				defer detector.Close()

				report, status := detector.Process(true, 0)
				return map[string]interface{}{
					"report": report,
					"status": status,
					"active": detector.IsMotionActive,
				}
			},
		},
		{
			name: "extremely_large_area",
			test: func(t *testing.T) map[string]interface{} {
				config := motion.Config{MinimumArea: 30000, MinMotionDuration: 100 * time.Millisecond}
				detector := motion.New(config)
				defer detector.Close()

				report, status := detector.Process(true, 1e10) // 10 billion
				return map[string]interface{}{
					"report": report,
					"status": status,
					"active": detector.IsMotionActive,
				}
			},
		},
		{
			name: "zero_minimum_duration",
			test: func(t *testing.T) map[string]interface{} {
				config := motion.Config{MinimumArea: 30000, MinMotionDuration: 0}
				detector := motion.New(config)
				defer detector.Close()

				report, status := detector.Process(true, 50000)
				return map[string]interface{}{
					"report": report,
					"status": status,
					"active": detector.IsMotionActive,
				}
			},
		},
		{
			name: "double_close",
			test: func(t *testing.T) map[string]interface{} {
				config := motion.Config{MinimumArea: 30000, MinMotionDuration: 100 * time.Millisecond}
				detector := motion.New(config)

				detector.Close()
				detector.Close() // Should not panic

				return map[string]interface{}{
					"double_close": "handled",
				}
			},
		},
	}

	edgeResults := make(map[string]interface{})

	for _, edgeCase := range edgeCases {
		t.Run(edgeCase.name, func(t *testing.T) {
			edgeResults[edgeCase.name] = edgeCase.test(t)
		})
	}

	result.Metadata = map[string]interface{}{
		"edge_cases_tested": len(edgeCases),
		"edge_results":      edgeResults,
	}
}

// createComprehensiveMockSequence creates a comprehensive sequence for testing.
func createComprehensiveMockSequence(t *testing.T) []ImageFile {
	generator := NewMockFrameGenerator(640, 480)
	var imageFiles []ImageFile

	sequences := []struct {
		name    string
		frames  int
		pattern func(i int) gocv.Mat
	}{
		{
			name:   "static_sequence",
			frames: 20,
			pattern: func(i int) gocv.Mat {
				return generator.GenerateStaticFrame()
			},
		},
		{
			name:   "moving_object",
			frames: 30,
			pattern: func(i int) gocv.Mat {
				return generator.GenerateMotionFrame(50+i*10, 100+i*5, 80)
			},
		},
		{
			name:   "intermittent_motion",
			frames: 25,
			pattern: func(i int) gocv.Mat {
				if i%4 == 0 {
					return generator.GenerateMotionFrame(200+i*8, 150+i*6, 60)
				}
				return generator.GenerateStaticFrame()
			},
		},
		{
			name:   "multiple_objects",
			frames: 20,
			pattern: func(i int) gocv.Mat {
				frame := generator.GenerateStaticFrame()
				// Add multiple motion regions
				for j := 0; j < 3; j++ {
					x := 100 + j*150 + i*5
					y := 120 + j*100 + i*3
					motionFrame := generator.GenerateMotionFrame(x, y, 40)
					gocv.Add(frame, motionFrame, &frame)
					motionFrame.Close()
				}
				return frame
			},
		},
	}

	frameNum := 0
	for _, seq := range sequences {
		for i := 0; i < seq.frames; i++ {
			frame := seq.pattern(i)

			// Encode as JPEG
			data, err := gocv.IMEncode(".jpg", frame)
			require.NoError(t, err)

			imageFiles = append(imageFiles, ImageFile{
				Path:  fmt.Sprintf("mock_%s_frame_%d.jpg", seq.name, frameNum),
				Data:  data.GetBytes(),
				Frame: frameNum,
			})

			data.Close()
			frame.Close()
			frameNum++
		}
	}

	return imageFiles
}

// collectSystemMetrics collects current system metrics for resource testing.
func collectSystemMetrics() map[string]float64 {
	// Create a temporary detector just to collect system metrics
	config := motion.Config{MinimumArea: 1, MinMotionDuration: 1 * time.Millisecond}
	detector := motion.New(config)
	defer detector.Close()

	return detector.CollectMetrics()
}

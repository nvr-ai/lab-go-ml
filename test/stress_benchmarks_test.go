package test

import (
	"image"
	"image/color"
	"runtime"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/motion"
	"gocv.io/x/gocv"
)

// BenchmarkLongRunningDetection simulates long-running motion detection scenarios.
func BenchmarkLongRunningDetection(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1280, 720)

	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 500 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	// Warm up background model with static frames
	for i := 0; i < 30; i++ {
		frame := generator.GenerateStaticFrame()
		segmenter.SegmentMotion(frame)
		frame.Close()
	}

	var startMem, endMem runtime.MemStats
	runtime.ReadMemStats(&startMem)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate realistic motion patterns: bursts of motion followed by stillness
		var frame gocv.Mat
		if i%100 < 20 { // 20% motion frames in bursts
			frame = generator.GenerateMotionFrame(
				100+i%500, 100+i%400, 50+i%100)
		} else {
			frame = generator.GenerateStaticFrame()
		}

		contours := segmenter.SegmentMotion(frame)
		
		totalArea := 0.0
		for j := 0; j < contours.Size(); j++ {
			area := gocv.ContourArea(contours.At(j))
			totalArea += area
		}
		
		hasMotion := totalArea > detector.MinimumArea
		detector.FPS(hasMotion)
		report, _ := detector.Process(hasMotion, totalArea)
		
		// Simulate some processing on reported motion
		if report {
			_ = detector.CollectMetrics()
		}

		contours.Close()
		frame.Close()

		// Periodic GC hint to test memory stability
		if i%1000 == 0 {
			runtime.GC()
		}
	}

	runtime.ReadMemStats(&endMem)

	result := &TestResult{
		TestName:  "BenchmarkLongRunningDetection",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":          b.N,
			"ns_per_op":           b.Elapsed().Nanoseconds() / int64(b.N),
			"heap_start_mb":       float64(startMem.HeapAlloc) / 1024 / 1024,
			"heap_end_mb":         float64(endMem.HeapAlloc) / 1024 / 1024,
			"heap_delta_mb":       float64(endMem.HeapAlloc-startMem.HeapAlloc) / 1024 / 1024,
			"gc_cycles":           endMem.NumGC - startMem.NumGC,
			"motion_events":       detector.MotionEventCount,
			"fps_potential":       float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":           "long_running_stability",
		},
	}
	store.Save(result)
}

// BenchmarkHighMotionScenario tests performance under high motion conditions.
func BenchmarkHighMotionScenario(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1920, 1080)

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(5, 5)) // Larger kernel for high motion
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	config := motion.Config{
		MinimumArea:       10000, // Lower threshold for high sensitivity
		MinMotionDuration: 100 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Generate multiple motion regions to stress the system
		frame := generator.GenerateStaticFrame()
		
		// Add multiple motion blobs
		for j := 0; j < 5; j++ {
			x := (i*50 + j*200) % (1920 - 150)
			y := (i*30 + j*150) % (1080 - 150)
			motionRect := image.Rect(x, y, x+150, y+150)
			gocv.Rectangle(&frame, motionRect, color.RGBA{255, 255, 255, 0}, -1)
		}

		contours := segmenter.SegmentMotion(frame)
		
		totalArea := 0.0
		contoursCount := contours.Size()
		for j := 0; j < contoursCount; j++ {
			area := gocv.ContourArea(contours.At(j))
			totalArea += area
		}
		
		hasMotion := totalArea > detector.MinimumArea
		detector.FPS(hasMotion)
		detector.Process(hasMotion, totalArea)

		contours.Close()
		frame.Close()
	}

	result := &TestResult{
		TestName:  "BenchmarkHighMotionScenario",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1920x1080",
			"motion_regions":     5,
			"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":          "high_motion_stress",
		},
	}
	store.Save(result)
}

// BenchmarkNoiseResilience tests performance with noisy input frames.
func BenchmarkNoiseResilience(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1280, 720)

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		frame := generator.GenerateStaticFrame()
		
		// Add random noise to simulate real-world conditions
		noise := gocv.NewMatWithSize(720, 1280, gocv.MatTypeCV8UC1)
		gocv.RandN(&noise, gocv.NewScalar(0, 0, 0, 0), gocv.NewScalar(25, 0, 0, 0))
		gocv.Add(frame, noise, &frame)
		noise.Close()

		// Add motion in some frames
		if i%4 == 0 {
			motionRect := image.Rect(200+i%400, 150+i%300, 300+i%400, 250+i%300)
			gocv.Rectangle(&frame, motionRect, color.RGBA{200, 200, 200, 0}, -1)
		}

		contours := segmenter.SegmentMotion(frame)
		contours.Close()
		frame.Close()
	}

	result := &TestResult{
		TestName:  "BenchmarkNoiseResilience",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1280x720",
			"noise_level":        "0-50 intensity",
			"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":          "noise_resilience",
		},
	}
	store.Save(result)
}

// BenchmarkRapidSceneChanges tests performance during rapid scene transitions.
func BenchmarkRapidSceneChanges(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1280, 720)

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	config := motion.Config{
		MinimumArea:       20000,
		MinMotionDuration: 200 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var frame gocv.Mat
		
		// Simulate rapid scene changes every 10 frames
		if i%10 == 0 {
			// Complete scene change - fill with different intensity
			intensity := 50 + (i%4)*50
			frame = gocv.NewMatWithSize(720, 1280, gocv.MatTypeCV8UC1)
			frame.SetTo(gocv.NewScalar(float64(intensity), 0, 0, 0))
		} else {
			// Regular motion frame
			frame = generator.GenerateMotionFrame(100+i%600, 100+i%400, 80)
		}

		contours := segmenter.SegmentMotion(frame)
		
		totalArea := 0.0
		for j := 0; j < contours.Size(); j++ {
			area := gocv.ContourArea(contours.At(j))
			totalArea += area
		}
		
		hasMotion := totalArea > detector.MinimumArea
		detector.FPS(hasMotion)
		detector.Process(hasMotion, totalArea)

		contours.Close()
		frame.Close()
	}

	result := &TestResult{
		TestName:  "BenchmarkRapidSceneChanges",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "1280x720",
			"scene_change_freq":  "every 10 frames",
			"motion_events":      detector.MotionEventCount,
			"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":          "rapid_scene_changes",
		},
	}
	store.Save(result)
}

// BenchmarkProfilerOverhead measures the overhead of profiling during motion detection.
func BenchmarkProfilerOverhead(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	generator := NewMockFrameGenerator(1280, 720)

	// Test with profiler enabled
	config := motion.Config{
		MinimumArea:       30000,
		MinMotionDuration: 500 * time.Millisecond,
	}
	detector := motion.New(config)
	defer detector.Close()

	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	// Start profiler
	detector.Profiler.Start()
	defer detector.Profiler.Stop()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		frame := generator.GenerateMotionFrame(200+i%400, 200+i%300, 100)
		
		start := time.Now()
		contours := segmenter.SegmentMotion(frame)
		detector.FrameProcessingTime = time.Since(start)
		
		totalArea := 0.0
		for j := 0; j < contours.Size(); j++ {
			area := gocv.ContourArea(contours.At(j))
			totalArea += area
		}
		
		hasMotion := totalArea > detector.MinimumArea
		detector.FPS(hasMotion)
		detector.Process(hasMotion, totalArea)
		
		// Collect metrics periodically to test profiler overhead
		if i%10 == 0 {
			_ = detector.CollectMetrics()
		}

		contours.Close()
		frame.Close()
	}

	result := &TestResult{
		TestName:  "BenchmarkProfilerOverhead",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"profiler_enabled":   true,
			"metrics_frequency":  "every 10 frames",
			"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":          "profiler_overhead",
		},
	}
	store.Save(result)
}

// BenchmarkEdgeCaseFrames tests performance with edge case frame conditions.
func BenchmarkEdgeCaseFrames(b *testing.B) {
	store := NewTestResultStore("./benchmark_results")
	
	segmenter := images.NewMotionSegmenter()
	segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	defer segmenter.Kernel.Close()
	defer segmenter.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var frame gocv.Mat
		
		switch i % 6 {
		case 0:
			// All black frame
			frame = gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC1)
			frame.SetTo(gocv.NewScalar(0, 0, 0, 0))
		case 1:
			// All white frame
			frame = gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC1)
			frame.SetTo(gocv.NewScalar(255, 0, 0, 0))
		case 2:
			// Checkerboard pattern
			frame = gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC1)
			for y := 0; y < 480; y += 20 {
				for x := 0; x < 640; x += 20 {
					val := 255
					if (x/20+y/20)%2 == 0 {
						val = 0
					}
					rect := image.Rect(x, y, x+20, y+20)
					gocv.Rectangle(&frame, rect, color.RGBA{uint8(val), uint8(val), uint8(val), 0}, -1)
				}
			}
		case 3:
			// Single pixel motion
			frame = gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC1)
			frame.SetTo(gocv.NewScalar(128, 0, 0, 0))
			frame.SetUCharAt(240, 320, 255)
		case 4:
			// Gradient frame
			frame = gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC1)
			for y := 0; y < 480; y++ {
				intensity := uint8((y * 255) / 480)
				for x := 0; x < 640; x++ {
					frame.SetUCharAt(y, x, intensity)
				}
			}
		case 5:
			// Thin line motion
			frame = gocv.NewMatWithSize(480, 640, gocv.MatTypeCV8UC1)
			frame.SetTo(gocv.NewScalar(128, 0, 0, 0))
			gocv.Line(&frame, image.Pt(0, 240), image.Pt(640, 240), color.RGBA{255, 255, 255, 0}, 1)
		}

		contours := segmenter.SegmentMotion(frame)
		contours.Close()
		frame.Close()
	}

	result := &TestResult{
		TestName:  "BenchmarkEdgeCaseFrames",
		Timestamp: time.Now(),
		Duration:  time.Duration(b.Elapsed()),
		Success:   true,
		Metadata: map[string]interface{}{
			"operations":         b.N,
			"ns_per_op":          b.Elapsed().Nanoseconds() / int64(b.N),
			"resolution":         "640x480",
			"edge_cases":         "black,white,checkerboard,single_pixel,gradient,thin_line",
			"fps_potential":      float64(1e9) / float64(b.Elapsed().Nanoseconds()/int64(b.N)),
			"test_type":          "edge_case_frames",
		},
	}
	store.Save(result)
}
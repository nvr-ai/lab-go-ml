package motion

// import (
// 	"fmt"
// 	"image"
// 	"runtime"
// 	"strings"
// 	"time"

// 	"github.com/nvr-ai/go-ml/inference/detectors"
// 	"github.com/nvr-ai/go-ml/inference/providers"
// 	"github.com/nvr-ai/go-ml/profiler"
// 	"gocv.io/x/gocv"
// )

// const (
// 	// MinimumArea represents the minimum area threshold for motion detection.
// 	MinimumArea = 30000
// )

// // Config represents the configuration for the motion detector.
// type Config struct {
// 	MinimumArea       float64
// 	MinMotionDuration time.Duration
// }

// // Detector encapsulates the motion detection state and logic.
// type Detector struct {
// 	// Configuration
// 	MinimumArea       float64
// 	MinMotionDuration time.Duration

// 	// Motion tracking
// 	MotionStartTime  *time.Time
// 	LastMotionTime   time.Time
// 	IsMotionActive   bool
// 	MotionEventCount int

// 	// FPS tracking
// 	TotalFrames  int
// 	MotionFrames int
// 	FPSStartTime time.Time
// 	CurrentFPS   float64
// 	MotionFPS    float64

// 	// Performance: reusable kernel
// 	Kernel gocv.Mat

// 	// Profiling support
// 	FrameProcessingTime time.Duration
// 	Profiler            *profiler.RuntimeProfiler
// }

// // New creates a new motion detector with default settings.
// func New(config Config) *Detector {
// 	md := &Detector{
// 		MinimumArea:       config.MinimumArea,
// 		MinMotionDuration: config.MinMotionDuration,
// 		FPSStartTime:      time.Now(),
// 		Kernel:            gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3)),
// 	}

// 	// Initialize profiler.
// 	opts := profiler.ProfilingOptions{
// 		ReportInterval: 2 * time.Second,
// 		SampleInterval: 100 * time.Millisecond,
// 		MaxSamples:     600,
// 	}
// 	md.Profiler = profiler.NewRuntimeProfiler(opts)

// 	// Add motion detection specific metrics collector.
// 	md.Profiler.AddMetricsCollector(md)

// 	_, err := detectors.NewSession(detectors.Config{
// 		Provider:            providers.DefaultConfig(),
// 		InputShape:          image.Point{X: 416, Y: 416},
// 		ConfidenceThreshold: 0.5,
// 		NMSThreshold:        0.7,
// 		RelevantClasses:     []string{"person", "car", "truck", "bus", "motorcycle", "bicycle"},
// 	})
// 	if err != nil {
// 		// Check if it's a library not found error
// 		if strings.Contains(err.Error(), "ONNX Runtime library not found") {
// 			panic(err)
// 		}
// 		panic(err)
// 	}

// 	return md
// }

// // Close releases resources used by the motion detector.
// func (d *Detector) Close() {
// 	d.Kernel.Close()
// 	if d.Profiler != nil {
// 		d.Profiler.Stop()
// 	}
// }

// // FPS updates the FPS calculations.
// func (d *Detector) FPS(hasMotion bool) {
// 	d.TotalFrames++
// 	if hasMotion {
// 		d.MotionFrames++
// 	}

// 	elapsed := time.Since(d.FPSStartTime).Seconds()
// 	if elapsed >= 1.0 {
// 		d.CurrentFPS = float64(d.TotalFrames) / elapsed
// 		d.MotionFPS = float64(d.MotionFrames) / elapsed

// 		// Reset counters
// 		d.TotalFrames = 0
// 		d.MotionFrames = 0
// 		d.FPSStartTime = time.Now()
// 	}
// }

// // Process handles motion state transitions and duration checking.
// func (d *Detector) Process(detected bool, area float64) (report bool, status string) {
// 	now := time.Now()
// 	status = "Ready"

// 	if detected {
// 		// Motion detected in this frame
// 		if !d.IsMotionActive {
// 			// New motion started
// 			d.MotionStartTime = &now
// 			d.IsMotionActive = true
// 		}
// 		d.LastMotionTime = now

// 		// Check if motion has lasted long enough
// 		if d.MotionStartTime != nil {
// 			duration := now.Sub(*d.MotionStartTime)
// 			if duration >= d.MinMotionDuration {
// 				report = true
// 				status = fmt.Sprintf("Motion: %.1fs | Area: %.0f", duration.Seconds(), area)
// 			} else {
// 				status = fmt.Sprintf("Motion: %.1fs (min: %.1fs)", duration.Seconds(),
// d.MinMotionDuration.Seconds())
// 			}
// 		}
// 	} else {
// 		// No motion in this frame
// 		if d.IsMotionActive {
// 			// Check if motion has stopped (with some tolerance)
// 			if time.Since(d.LastMotionTime) > 100*time.Millisecond {
// 				// Motion ended
// 				if d.MotionStartTime != nil {
// 					duration := d.LastMotionTime.Sub(*d.MotionStartTime)
// 					if duration >= d.MinMotionDuration {
// 						d.MotionEventCount++
// 					}
// 				}
// 				d.IsMotionActive = false
// 				d.MotionStartTime = nil
// 			}
// 		}
// 	}

// 	return report, status
// }

// // CollectMetrics implements the MetricsCollector interface.
// //
// // Returns:
// // - A map of metric names to their current values
// func (d *Detector) CollectMetrics() map[string]float64 {
// 	metrics := make(map[string]float64)

// 	// Collect motion detection specific metrics
// 	metrics["fps_total"] = float64(d.TotalFrames)
// 	metrics["fps_motion"] = float64(d.MotionFrames)
// 	metrics["motion_events"] = float64(d.MotionEventCount)
// 	metrics["frame_processing_ms"] = float64(d.FrameProcessingTime.Nanoseconds()) / 1e6

// 	// Calculate frame interval
// 	now := time.Now()
// 	frameInterval := now.Sub(d.FPSStartTime)
// 	metrics["frame_interval_ms"] = float64(frameInterval.Nanoseconds()) / 1e6
// 	d.FPSStartTime = now

// 	// System-level metrics specific to CV workloads
// 	var m runtime.MemStats
// 	runtime.ReadMemStats(&m)

// 	metrics["heap_alloc_mb"] = float64(m.HeapAlloc) / 1024 / 1024
// 	metrics["goroutines"] = float64(runtime.NumGoroutine())

// 	return metrics
// }

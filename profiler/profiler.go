package profiler

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// MetricsCollector defines the interface for collecting custom metrics.
type MetricsCollector interface {
	CollectMetrics() map[string]float64
}

// RuntimeProfiler provides comprehensive runtime profiling and monitoring.
//
// The profiler tracks system resources, custom application metrics, and provides
// detailed periodic reports. It's designed to be thread-safe and easily integrated
// into any Go application.
type RuntimeProfiler struct {
	// Configuration
	reportInterval time.Duration
	sampleInterval time.Duration

	// State management
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	mu        sync.RWMutex
	startTime time.Time
	running   bool

	// System metrics
	memStats    runtime.MemStats
	cpuSamples  []cpuSample
	maxSamples  int
	gcCount     uint32
	lastGCCount uint32

	// Custom metrics
	customMetrics map[string]*MetricTracker
	collectors    []MetricsCollector

	// Performance tracking
	operationTimes map[string]*TimeTracker
}

// cpuSample represents a CPU usage sample.
type cpuSample struct {
	timestamp  time.Time
	goroutines int
	cgoCalls   int64
}

// MetricTracker tracks statistics for a custom metric.
type MetricTracker struct {
	name     string
	values   []float64
	sum      float64
	min      float64
	max      float64
	count    int64
	lastTime time.Time
}

// TimeTracker tracks operation timing statistics.
type TimeTracker struct {
	name      string
	durations []time.Duration
	totalTime time.Duration
	minTime   time.Duration
	maxTime   time.Duration
	count     int64
}

// ProfilingOptions configures the runtime profiler.
type ProfilingOptions struct {
	// ReportInterval specifies how often to emit status reports (default: 2s)
	ReportInterval time.Duration
	// SampleInterval specifies how often to collect samples (default: 100ms)
	SampleInterval time.Duration
	// MaxSamples specifies maximum number of samples to keep (default: 600)
	MaxSamples int
}

// NewRuntimeProfiler creates a new runtime profiler with the specified options.
//
// Arguments:
// - opts: Configuration options for the profiler
//
// Returns:
// - A configured RuntimeProfiler instance
func NewRuntimeProfiler(opts ProfilingOptions) *RuntimeProfiler {
	// Set defaults
	if opts.ReportInterval == 0 {
		opts.ReportInterval = 2 * time.Second
	}
	if opts.SampleInterval == 0 {
		opts.SampleInterval = 100 * time.Millisecond
	}
	if opts.MaxSamples == 0 {
		opts.MaxSamples = 600 // 1 minute of samples at 100ms intervals
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &RuntimeProfiler{
		reportInterval: opts.ReportInterval,
		sampleInterval: opts.SampleInterval,
		ctx:            ctx,
		cancel:         cancel,
		startTime:      time.Now(),
		maxSamples:     opts.MaxSamples,
		cpuSamples:     make([]cpuSample, 0, opts.MaxSamples),
		customMetrics:  make(map[string]*MetricTracker),
		collectors:     make([]MetricsCollector, 0),
		operationTimes: make(map[string]*TimeTracker),
	}
}

// Start begins the profiling and reporting process.
//
// This method starts background goroutines for collecting samples and emitting
// periodic reports. It's thread-safe and can be called multiple times safely.
func (rp *RuntimeProfiler) Start() {
	rp.mu.Lock()
	defer rp.mu.Unlock()

	if rp.running {
		return
	}

	rp.running = true
	rp.startTime = time.Now()

	// Start sampling goroutine.
	rp.wg.Add(1)
	go rp.sampleLoop()

	// Start reporting goroutine.
	rp.wg.Add(1)
	go func() {
		defer rp.wg.Done()

		ticker := time.NewTicker(rp.reportInterval)
		defer ticker.Stop()

		for {
			select {
			case <-rp.ctx.Done():
				return
			case <-ticker.C:
				rp.emitStatusReport()
			}
		}
	}()
}

// Stop gracefully stops the profiler and waits for all goroutines to complete.
func (rp *RuntimeProfiler) Stop() {
	rp.mu.Lock()
	if !rp.running {
		rp.mu.Unlock()
		return
	}
	rp.running = false
	rp.mu.Unlock()

	rp.cancel()
	rp.wg.Wait()
}

// AddMetricsCollector registers a custom metrics collector to be called
// periodically.
//
// Arguments:
// - collector: An implementation of MetricsCollector interface
func (rp *RuntimeProfiler) AddMetricsCollector(collector MetricsCollector) {
	rp.mu.Lock()
	defer rp.mu.Unlock()
	rp.collectors = append(rp.collectors, collector)
}

// RecordMetric records a custom metric value.
//
// Arguments:
// - name: The name of the metric
// - value: The metric value to record
func (rp *RuntimeProfiler) RecordMetric(name string, value float64) {
	rp.mu.Lock()
	defer rp.mu.Unlock()

	tracker, exists := rp.customMetrics[name]
	if !exists {
		tracker = &MetricTracker{
			name:   name,
			values: make([]float64, 0, rp.maxSamples),
			min:    value,
			max:    value,
		}
		rp.customMetrics[name] = tracker
	}

	// Update statistics
	tracker.values = append(tracker.values, value)
	if len(tracker.values) > rp.maxSamples {
		// Remove oldest sample
		tracker.sum -= tracker.values[0]
		tracker.values = tracker.values[1:]
	}

	tracker.sum += value
	tracker.count++
	tracker.lastTime = time.Now()

	if value < tracker.min {
		tracker.min = value
	}
	if value > tracker.max {
		tracker.max = value
	}
}

// StartOperation begins timing an operation.
//
// Arguments:
// - name: The name of the operation to track
//
// Returns:
// - A function to call when the operation completes
func (rp *RuntimeProfiler) StartOperation(name string) func() {
	start := time.Now()
	return func() {
		duration := time.Since(start)
		rp.recordOperationTime(name, duration)
	}
}

// recordOperationTime records the completion time of an operation.
func (rp *RuntimeProfiler) recordOperationTime(name string, duration time.Duration) {
	rp.mu.Lock()
	defer rp.mu.Unlock()

	tracker, exists := rp.operationTimes[name]
	if !exists {
		tracker = &TimeTracker{
			name:    name,
			minTime: duration,
			maxTime: duration,
		}
		rp.operationTimes[name] = tracker
	}

	tracker.durations = append(tracker.durations, duration)
	if len(tracker.durations) > rp.maxSamples {
		// Remove oldest sample
		tracker.totalTime -= tracker.durations[0]
		tracker.durations = tracker.durations[1:]
	}

	tracker.totalTime += duration
	tracker.count++

	if duration < tracker.minTime {
		tracker.minTime = duration
	}
	if duration > tracker.maxTime {
		tracker.maxTime = duration
	}
}

// sampleLoop continuously collects system metrics.
func (rp *RuntimeProfiler) sampleLoop() {
	defer rp.wg.Done()

	ticker := time.NewTicker(rp.sampleInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rp.ctx.Done():
			return
		case <-ticker.C:
			rp.mu.Lock()
			defer rp.mu.Unlock()

			runtime.ReadMemStats(&rp.memStats)

			rp.cpuSamples = append(rp.cpuSamples, cpuSample{
				timestamp:  time.Now(),
				goroutines: runtime.NumGoroutine(),
				cgoCalls:   runtime.NumCgoCall(),
			})
			if len(rp.cpuSamples) > rp.maxSamples {
				rp.cpuSamples = rp.cpuSamples[1:]
			}

			// Collect custom metrics from registered collectors.
			for _, collector := range rp.collectors {
				metrics := collector.CollectMetrics()
				for name, value := range metrics {
					tracker, exists := rp.customMetrics[name]
					if !exists {
						tracker = &MetricTracker{
							name:   name,
							values: make([]float64, 0, rp.maxSamples),
							min:    value,
							max:    value,
						}
						rp.customMetrics[name] = tracker
					}

					tracker.values = append(tracker.values, value)

					if len(tracker.values) > rp.maxSamples {
						tracker.sum -= tracker.values[0]
						tracker.values = tracker.values[1:]
					}

					tracker.sum += value
					tracker.count++

					if value < tracker.min {
						tracker.min = value
					}

					if value > tracker.max {
						tracker.max = value
					}

					tracker.lastTime = time.Now()
				}
			}
		}
	}
}

// emitStatusReport generates and prints a comprehensive status report.
func (rp *RuntimeProfiler) emitStatusReport() {
	rp.mu.RLock()
	defer rp.mu.RUnlock()

	uptime := time.Since(rp.startTime)

	fmt.Printf("RUNTIME PROFILER STATUS REPORT - %s\n", time.Now().Format("15:04:05.000"))
	fmt.Printf("Uptime: %v\n", uptime.Truncate(time.Millisecond))

	// System metrics
	fmt.Printf("\nSYSTEM METRICS:\n")
	fmt.Printf("  Goroutines: %d\n", runtime.NumGoroutine())
	fmt.Printf("  CGO Calls: %d\n", runtime.NumCgoCall())

	// Memory statistics
	fmt.Printf("\nMEMORY USAGE:\n")
	fmt.Printf("  Alloc: %s\n", formatBytes(rp.memStats.Alloc))
	fmt.Printf("  Total Alloc: %s\n", formatBytes(rp.memStats.TotalAlloc))
	fmt.Printf("  Sys: %s\n", formatBytes(rp.memStats.Sys))
	fmt.Printf("  Heap Alloc: %s\n", formatBytes(rp.memStats.HeapAlloc))
	fmt.Printf("  Heap Sys: %s\n", formatBytes(rp.memStats.HeapSys))
	fmt.Printf("  Heap Objects: %d\n", rp.memStats.HeapObjects)

	// GC statistics
	if rp.memStats.NumGC > rp.lastGCCount {
		fmt.Printf("\nGARBAGE COLLECTION:\n")
		fmt.Printf("  GC Cycles: %d (new: %d)\n", rp.memStats.NumGC, rp.memStats.NumGC-rp.lastGCCount)
		fmt.Printf("  Last GC: %v ago\n", time.Since(time.Unix(0, int64(rp.memStats.LastGC))).Truncate(time.Millisecond))
		fmt.Printf("  GC CPU Fraction: %.4f%%\n", rp.memStats.GCCPUFraction*100)
		rp.lastGCCount = rp.memStats.NumGC
	}

	// Custom metrics
	if len(rp.customMetrics) > 0 {
		fmt.Printf("\nCUSTOM METRICS:\n")
		for name, tracker := range rp.customMetrics {
			if len(tracker.values) > 0 {
				avg := tracker.sum / float64(len(tracker.values))
				fmt.Printf("  %s: avg=%.2f, min=%.2f, max=%.2f, samples=%d\n",
					name, avg, tracker.min, tracker.max, len(tracker.values))
			}
		}
	}

	// Operation timings
	if len(rp.operationTimes) > 0 {
		fmt.Printf("\nOPERATION TIMINGS:\n")
		for name, tracker := range rp.operationTimes {
			if len(tracker.durations) > 0 {
				avgTime := tracker.totalTime / time.Duration(len(tracker.durations))
				fmt.Printf("  %s: avg=%v, min=%v, max=%v, count=%d\n",
					name, avgTime.Truncate(time.Microsecond),
					tracker.minTime.Truncate(time.Microsecond),
					tracker.maxTime.Truncate(time.Microsecond),
					len(tracker.durations))
			}
		}
	}
}

// formatBytes formats byte counts in human-readable format.
func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// GetCurrentStats returns the current profiling statistics as a snapshot.
//
// Returns:
// - A map containing current statistics and metrics
func (rp *RuntimeProfiler) GetCurrentStats() map[string]interface{} {
	rp.mu.RLock()
	defer rp.mu.RUnlock()

	stats := make(map[string]interface{})

	// System stats
	stats["uptime"] = time.Since(rp.startTime)
	stats["goroutines"] = runtime.NumGoroutine()
	stats["cgo_calls"] = runtime.NumCgoCall()

	// Memory stats
	runtime.ReadMemStats(&rp.memStats)
	stats["memory"] = map[string]interface{}{
		"alloc":           rp.memStats.Alloc,
		"total_alloc":     rp.memStats.TotalAlloc,
		"sys":             rp.memStats.Sys,
		"heap_alloc":      rp.memStats.HeapAlloc,
		"heap_sys":        rp.memStats.HeapSys,
		"heap_objects":    rp.memStats.HeapObjects,
		"gc_cycles":       rp.memStats.NumGC,
		"gc_cpu_fraction": rp.memStats.GCCPUFraction,
	}

	// Custom metrics
	customStats := make(map[string]interface{})
	for name, tracker := range rp.customMetrics {
		if len(tracker.values) > 0 {
			avg := tracker.sum / float64(len(tracker.values))
			customStats[name] = map[string]interface{}{
				"avg":     avg,
				"min":     tracker.min,
				"max":     tracker.max,
				"samples": len(tracker.values),
			}
		}
	}
	stats["custom_metrics"] = customStats

	return stats
}

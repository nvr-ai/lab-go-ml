package kernels

import (
	"fmt"
	"image"
	"runtime"
	"runtime/debug"
	"time"
)

// MemoryProfiler provides detailed memory usage analysis for blur operations
// Critical for understanding GC impact in real-time video processing pipelines
type MemoryProfiler struct {
	samples          []MemorySample   // Time series of memory measurements
	gcStats          []GCEvent        // Garbage collection events
	baseline         runtime.MemStats // Initial memory state
	enableDetailedGC bool             // Track individual GC events
	samplingInterval time.Duration    // How often to sample memory
}

// MemorySample captures memory state at a specific point in time
type MemorySample struct {
	Timestamp    time.Time // When the sample was taken
	HeapAlloc    uint64    // Currently allocated heap memory (bytes)
	HeapSys      uint64    // Total heap memory from OS (bytes)
	HeapInuse    uint64    // In-use heap memory (bytes)
	HeapReleased uint64    // Heap memory returned to OS (bytes)
	StackInuse   uint64    // Stack memory in use (bytes)
	Mallocs      uint64    // Cumulative allocations count
	Frees        uint64    // Cumulative frees count
	PauseTotalNs uint64    // Cumulative GC pause time (nanoseconds)
	NumGC        uint32    // Number of completed GC cycles

	// Derived metrics for easier analysis
	LiveObjects     uint64  // MalCls - Frees (approximate live object count)
	HeapUtilization float64 // HeapInuse / HeapSys (how efficiently heap is used)
	AllocationRate  float64 // Bytes/second allocation rate (calculated)
}

// GCEvent captures details about individual garbage collection cycles
type GCEvent struct {
	Timestamp     time.Time     // When GC started
	PauseDuration time.Duration // Stop-the-world pause time
	HeapBefore    uint64        // Heap size before GC
	HeapAfter     uint64        // Heap size after GC
	GCType        string        // GC algorithm used ("mark-sweep", "concurrent", etc.)
	TriggerReason string        // What triggered this GC cycle
}

// NewMemoryProfiler creates a memory profiler with specified sampling rate
func NewMemoryProfiler(samplingInterval time.Duration, enableDetailedGC bool) *MemoryProfiler {
	profiler := &MemoryProfiler{
		samples:          make([]MemorySample, 0, 1000), // Pre-allocate for efficiency
		gcStats:          make([]GCEvent, 0, 100),
		samplingInterval: samplingInterval,
		enableDetailedGC: enableDetailedGC,
	}

	// Capture baseline memory state
	runtime.ReadMemStats(&profiler.baseline)

	if enableDetailedGC {
		// Set up detailed GC tracking
		debug.SetGCPercent(100) // Default GC trigger threshold
		profiler.setupGCTracking()
	}

	return profiler
}

// setupGCTracking configures detailed GC event monitoring
func (mp *MemoryProfiler) setupGCTracking() {
	// Note: This is a simplified GC tracking setup
	// In production, you might use more sophisticated tools like
	// runtime/trace or external profilers like pprof

	// For this implementation, we'll track GC through periodic polling
	// Real-time GC event tracking would require runtime hooks
}

// StartProfiling begins continuous memory monitoring
// Should be called in a separate goroutine for non-blocking operation
func (mp *MemoryProfiler) StartProfiling() {
	go func() {
		ticker := time.NewTicker(mp.samplingInterval)
		defer ticker.Stop()

		var lastSample MemorySample
		// startTime := time.Now()

		for range ticker.C {
			sample := mp.captureSample()

			// Calculate allocation rate if we have a previous sample
			if len(mp.samples) > 0 {
				timeDelta := sample.Timestamp.Sub(lastSample.Timestamp).Seconds()
				if timeDelta > 0 {
					allocDelta := sample.HeapAlloc - lastSample.HeapAlloc
					sample.AllocationRate = float64(allocDelta) / timeDelta
				}
			}

			mp.samples = append(mp.samples, sample)
			lastSample = sample

			// Detect and record GC events
			if mp.enableDetailedGC && len(mp.samples) > 1 {
				mp.detectGCEvents(mp.samples[len(mp.samples)-2], sample)
			}

			// Prevent unbounded memory growth in long-running profiles
			if len(mp.samples) > 10000 {
				// Keep recent samples, discard old ones
				copy(mp.samples, mp.samples[5000:])
				mp.samples = mp.samples[:5000]
			}
		}
	}()
}

// captureSample takes a snapshot of current memory statistics
func (mp *MemoryProfiler) captureSample() MemorySample {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	sample := MemorySample{
		Timestamp:       time.Now(),
		HeapAlloc:       m.HeapAlloc,
		HeapSys:         m.HeapSys,
		HeapInuse:       m.HeapInuse,
		HeapReleased:    m.HeapReleased,
		StackInuse:      m.StackInuse,
		Mallocs:         m.Mallocs,
		Frees:           m.Frees,
		PauseTotalNs:    m.PauseTotalNs,
		NumGC:           m.NumGC,
		LiveObjects:     m.Mallocs - m.Frees,
		HeapUtilization: float64(m.HeapInuse) / float64(m.HeapSys),
	}

	return sample
}

// detectGCEvents identifies garbage collection cycles between samples
func (mp *MemoryProfiler) detectGCEvents(prev, curr MemorySample) {
	if curr.NumGC > prev.NumGC {
		// One or more GC cycles occurred
		gcCount := curr.NumGC - prev.NumGC
		avgPauseDuration := time.Duration((curr.PauseTotalNs - prev.PauseTotalNs) / uint64(gcCount))

		// Create GC event record
		// Note: This is simplified - real GC tracking would provide more details
		event := GCEvent{
			Timestamp:     curr.Timestamp,
			PauseDuration: avgPauseDuration,
			HeapBefore:    prev.HeapAlloc,
			HeapAfter:     curr.HeapAlloc,
			GCType:        "mark-sweep", // Go's standard algorithm
			TriggerReason: "heap-size",  // Most common trigger
		}

		mp.gcStats = append(mp.gcStats, event)
	}
}

// StopProfiling ends memory monitoring and returns analysis results
func (mp *MemoryProfiler) StopProfiling() *MemoryAnalysisReport {
	// Generate comprehensive analysis report
	return mp.generateReport()
}

// MemoryAnalysisReport contains detailed memory usage analysis
type MemoryAnalysisReport struct {
	// Summary statistics
	Duration         time.Duration `json:"duration"`           // Total profiling duration
	TotalAllocations uint64        `json:"total_allocations"`  // Total bytes allocated
	PeakHeapUsage    uint64        `json:"peak_heap_usage"`    // Maximum heap memory used
	AverageHeapUsage uint64        `json:"average_heap_usage"` // Mean heap memory used
	TotalGCPauses    time.Duration `json:"total_gc_pauses"`    // Cumulative GC pause time
	GCFrequency      float64       `json:"gc_frequency"`       // GC cycles per second

	// Performance impact metrics
	GCOverhead       float64 `json:"gc_overhead"`       // GC time as % of total time
	AllocationRate   float64 `json:"allocation_rate"`   // Average bytes/second allocated
	MemoryEfficiency float64 `json:"memory_efficiency"` // HeapInuse / HeapSys ratio

	// Detailed breakdowns
	GCEvents       []GCEvent     `json:"gc_events"`        // Individual GC occurrences
	SampleCount    int           `json:"sample_count"`     // Number of memory samples
	LargestGCPause time.Duration `json:"largest_gc_pause"` // Worst-case GC latency

	// Video processing specific metrics
	FrameProcessingImpact float64 `json:"frame_processing_impact"` // Estimated FPS reduction due to GC
	RecommendedPoolSize   uint64  `json:"recommended_pool_size"`   // Suggested memory pool allocation
}

// generateReport creates comprehensive analysis from collected samples
func (mp *MemoryProfiler) generateReport() *MemoryAnalysisReport {
	if len(mp.samples) == 0 {
		return &MemoryAnalysisReport{}
	}

	first := mp.samples[0]
	last := mp.samples[len(mp.samples)-1]
	duration := last.Timestamp.Sub(first.Timestamp)

	// Calculate summary statistics
	var totalHeap, peakHeap uint64
	var totalAllocationRate float64

	for _, sample := range mp.samples {
		totalHeap += sample.HeapAlloc
		if sample.HeapAlloc > peakHeap {
			peakHeap = sample.HeapAlloc
		}
		totalAllocationRate += sample.AllocationRate
	}

	avgHeap := totalHeap / uint64(len(mp.samples))
	avgAllocationRate := totalAllocationRate / float64(len(mp.samples))

	// GC analysis
	totalPauses := time.Duration(last.PauseTotalNs - first.PauseTotalNs)
	gcCount := last.NumGC - first.NumGC
	gcFrequency := float64(gcCount) / duration.Seconds()
	gcOverhead := float64(totalPauses) / float64(duration)

	// Find largest GC pause
	var largestPause time.Duration
	for _, event := range mp.gcStats {
		if event.PauseDuration > largestPause {
			largestPause = event.PauseDuration
		}
	}

	// Estimate impact on video processing
	// Assumes 30 FPS target, calculates estimated frame drops due to GC pauses
	targetFrameTime := time.Second / 30
	frameProcessingImpact := float64(largestPause) / float64(targetFrameTime)

	// Recommend memory pool size based on peak usage patterns
	recommendedPoolSize := peakHeap * 2 // 2x peak usage for safety margin

	return &MemoryAnalysisReport{
		Duration:              duration,
		TotalAllocations:      last.HeapAlloc - first.HeapAlloc,
		PeakHeapUsage:         peakHeap,
		AverageHeapUsage:      avgHeap,
		TotalGCPauses:         totalPauses,
		GCFrequency:           gcFrequency,
		GCOverhead:            gcOverhead,
		AllocationRate:        avgAllocationRate,
		MemoryEfficiency:      float64(last.HeapInuse) / float64(last.HeapSys),
		GCEvents:              mp.gcStats,
		SampleCount:           len(mp.samples),
		LargestGCPause:        largestPause,
		FrameProcessingImpact: frameProcessingImpact,
		RecommendedPoolSize:   recommendedPoolSize,
	}
}

// ProfileBlurOperation performs detailed memory analysis of a blur operation
func ProfileBlurOperation(img image.Image, opts Options, iterations int) *BlurMemoryProfile {
	profiler := NewMemoryProfiler(10*time.Millisecond, true)
	profiler.StartProfiling()

	// Capture pre-operation state
	var m1, m2 runtime.MemStats
	runtime.GC() // Clean slate
	runtime.ReadMemStats(&m1)

	startTime := time.Now()

	// Execute blur operations
	for i := 0; i < iterations; i++ {
		result := BoxBlur(img, opts)
		_ = result // Prevent optimization elimination

		// Optionally trigger GC periodically to measure impact
		if i%10 == 0 {
			runtime.GC()
		}
	}

	endTime := time.Now()
	runtime.ReadMemStats(&m2)

	// Stop profiling and get detailed report
	report := profiler.StopProfiling()

	return &BlurMemoryProfile{
		MemoryAnalysisReport: report,
		OperationDuration:    endTime.Sub(startTime),
		Iterations:        iterations,
		ImageSize:         img.Bounds().Size(),
		BlurRadius:        opts.Radius,
		PoolEnabled:       opts.Pool != nil,

		// Per-iteration metrics
		AllocationsPerIteration: (m2.Mallocs - m1.Mallocs) / uint64(iterations),
		BytesPerIteration:       (m2.TotalAlloc - m1.TotalAlloc) / uint64(iterations),
		AvgIterationTime:        endTime.Sub(startTime) / time.Duration(iterations),
	}
}

// BlurMemoryProfile contains memory analysis specific to blur operations
type BlurMemoryProfile struct {
	*MemoryAnalysisReport

	// Operation-specific metrics
	OperationDuration time.Duration `json:"operation_duration"`
	Iterations        int           `json:"iterations"`
	ImageSize         image.Point   `json:"image_size"`
	BlurRadius        int           `json:"blur_radius"`
	PoolEnabled       bool          `json:"pool_enabled"`

	// Per-iteration breakdown
	AllocationsPerIteration uint64        `json:"allocations_per_iteration"`
	BytesPerIteration       uint64        `json:"bytes_per_iteration"`
	AvgIterationTime        time.Duration `json:"avg_iteration_time"`
}

// CompareMemoryProfiles compares two blur implementations' memory characteristics
func CompareMemoryProfiles(profile1, profile2 *BlurMemoryProfile, name1, name2 string) *MemoryComparisonReport {
	return &MemoryComparisonReport{
		Implementation1: name1,
		Implementation2: name2,
		Profile1:        profile1,
		Profile2:        profile2,

		// Comparative metrics
		AllocationRatio:  float64(profile1.BytesPerIteration) / float64(profile2.BytesPerIteration),
		GCFrequencyRatio: profile1.GCFrequency / profile2.GCFrequency,
		PeakMemoryRatio:  float64(profile1.PeakHeapUsage) / float64(profile2.PeakHeapUsage),
		GCOverheadRatio:  profile1.GCOverhead / profile2.GCOverhead,
		PerformanceRatio: profile2.AvgIterationTime.Nanoseconds() / profile1.AvgIterationTime.Nanoseconds(),

		// Recommendations
		RecommendedChoice: determineRecommendation(profile1, profile2, name1, name2),
	}
}

// MemoryComparisonReport provides side-by-side memory usage comparison
type MemoryComparisonReport struct {
	Implementation1 string             `json:"implementation_1"`
	Implementation2 string             `json:"implementation_2"`
	Profile1        *BlurMemoryProfile `json:"profile_1"`
	Profile2        *BlurMemoryProfile `json:"profile_2"`

	// Comparative ratios (Profile1 / Profile2)
	AllocationRatio  float64 `json:"allocation_ratio"`   // Memory allocation efficiency
	GCFrequencyRatio float64 `json:"gc_frequency_ratio"` // GC pressure comparison
	PeakMemoryRatio  float64 `json:"peak_memory_ratio"`  // Peak usage comparison
	GCOverheadRatio  float64 `json:"gc_overhead_ratio"`  // GC impact comparison
	PerformanceRatio int64   `json:"performance_ratio"`  // Speed comparison

	// Analysis and recommendations
	RecommendedChoice    string   `json:"recommended_choice"`
	JustificationReasons []string `json:"justification_reasons"`
}

// determineRecommendation analyzes profiles and provides implementation recommendation
func determineRecommendation(p1, p2 *BlurMemoryProfile, name1, name2 string) string {
	// Scoring system based on multiple factors
	score1 := calculateProfileScore(p1)
	score2 := calculateProfileScore(p2)

	if score1 > score2 {
		return name1
	} else if score2 > score1 {
		return name2
	} else {
		return "equivalent"
	}
}

// calculateProfileScore assigns a composite score based on memory efficiency
func calculateProfileScore(profile *BlurMemoryProfile) float64 {
	score := 1000.0 // Base score

	// Penalize high allocation rates (bytes/iteration)
	allocPenalty := float64(profile.BytesPerIteration) / (1024 * 1024) // MB per iteration
	score -= allocPenalty * 100

	// Penalize frequent GC (cycles/second)
	gcPenalty := profile.GCFrequency * 50
	score -= gcPenalty

	// Penalize high GC overhead (% of time in GC)
	overheadPenalty := profile.GCOverhead * 500
	score -= overheadPenalty

	// Penalize long GC pauses (milliseconds)
	pausePenalty := float64(profile.LargestGCPause.Nanoseconds()) / 1e6 / 10
	score -= pausePenalty

	// Reward fast execution (bonus for speed)
	speedBonus := 1000.0 / float64(profile.AvgIterationTime.Nanoseconds()) * 1e6
	score += speedBonus

	// Reward memory pool usage (if enabled)
	if profile.PoolEnabled {
		score += 200 // Significant bonus for using memory pools
	}

	return score
}

// FormatReport generates human-readable memory analysis report
func (report *MemoryAnalysisReport) FormatReport() string {
	return fmt.Sprintf(`
Memory Analysis Report
======================
Duration: %v
Samples Collected: %d

Memory Usage:
  Peak Heap: %.2f MB
  Average Heap: %.2f MB  
  Total Allocated: %.2f MB
  Allocation Rate: %.2f MB/s
  Memory Efficiency: %.1f%%

Garbage Collection:
  Total GC Time: %v
  GC Frequency: %.2f cycles/sec
  GC Overhead: %.2f%% of execution time
  Largest Pause: %v
  Number of GC Events: %d

Video Processing Impact:
  Estimated Frame Drops: %.2f%% 
  Recommended Pool Size: %.2f MB
`,
		report.Duration,
		report.SampleCount,
		float64(report.PeakHeapUsage)/1024/1024,
		float64(report.AverageHeapUsage)/1024/1024,
		float64(report.TotalAllocations)/1024/1024,
		report.AllocationRate/1024/1024,
		report.MemoryEfficiency*100,
		report.TotalGCPauses,
		report.GCFrequency,
		report.GCOverhead*100,
		report.LargestGCPause,
		len(report.GCEvents),
		report.FrameProcessingImpact*100,
		float64(report.RecommendedPoolSize)/1024/1024)
}

// FormatComparison generates human-readable comparison between two implementations
func (comparison *MemoryComparisonReport) FormatComparison() string {
	winner := comparison.RecommendedChoice
	var advantages []string

	if comparison.AllocationRatio < 1.0 {
		advantages = append(advantages, fmt.Sprintf("%.1fx less memory allocation", 1.0/comparison.AllocationRatio))
	}
	if comparison.GCFrequencyRatio < 1.0 {
		advantages = append(advantages, fmt.Sprintf("%.1fx less GC pressure", 1.0/comparison.GCFrequencyRatio))
	}
	if comparison.PerformanceRatio > 1 {
		advantages = append(advantages, fmt.Sprintf("%.1fx faster execution", float64(comparison.PerformanceRatio)))
	}

	return fmt.Sprintf(`
Memory Comparison Report
========================
%s vs %s

Winner: %s

Key Advantages:
%s

Detailed Ratios (%s / %s):
  Memory Allocation: %.2fx
  GC Frequency: %.2fx  
  Peak Memory: %.2fx
  GC Overhead: %.2fx
  Performance: %.1fx
`,
		comparison.Implementation1,
		comparison.Implementation2,
		winner,
		joinStrings(advantages, "\n  - "),
		comparison.Implementation1,
		comparison.Implementation2,
		comparison.AllocationRatio,
		comparison.GCFrequencyRatio,
		comparison.PeakMemoryRatio,
		comparison.GCOverheadRatio,
		float64(comparison.PerformanceRatio))
}

// Helper function to join strings with separator
func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return "  - None identified"
	}
	result := ""
	for i, s := range strs {
		if i > 0 {
			result += sep
		}
		result += "  - " + s
	}
	return result
}

package benchmark

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/inference"
	"github.com/nvr-ai/go-ml/inference/providers"
	"github.com/nvr-ai/go-ml/util"
)

// Suite manages and executes benchmark scenarios
type Suite struct {
	scenarios   []Scenario
	engine      inference.Engine
	outputDir   string
	corpus      []util.ImageFile
	imageFormat images.ImageFormat
	mu          sync.RWMutex
	results     []PerformanceMetrics
}

// NewSuite creates a new benchmark suite
func NewSuite(outputDir string) *Suite {
	engine := providers.New
	engine.LoadModel(*modelPath, nil)

	return &Suite{
		engine:    engine,
		outputDir: outputDir,
		scenarios: make([]Scenario, 0),
		results:   make([]PerformanceMetrics, 0),
	}
}

// AddScenario adds a test scenario to the benchmark suite
func (bs *Suite) AddScenario(scenario Scenario) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.scenarios = append(bs.scenarios, scenario)
}

// RunScenario executes a single benchmark scenario
func (bs *Suite) RunScenario(ctx context.Context, scenario Scenario) (*PerformanceMetrics, error) {
	// Load model
	modelConfig := map[string]interface{}{
		"input_shape":          []int{scenario.Resolution.Width, scenario.Resolution.Height},
		"confidence_threshold": 0.5,
		"nms_threshold":        0.4,
	}

	if err := bs.engine.LoadModel(scenario.ModelPath, modelConfig); err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	defer bs.engine.Close()

	metrics := &PerformanceMetrics{
		Scenario:  scenario,
		Timestamp: time.Now(),
	}

	// Warmup runs
	for i := 0; i < scenario.WarmupRuns; i++ {
		if len(bs.corpus) > 0 {
			testImg := bs.corpus[i%len(bs.corpus)]
			if _, err := bs.processImage(ctx, testImg, scenario); err != nil {
				continue // Skip warmup errors
			}
		}
	}

	// Capture initial memory stats
	var startMem runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&startMem)

	startTime := time.Now()
	totalDetections := 0
	errors := 0

	// Run benchmark iterations
	for i := 0; i < scenario.Iterations; i++ {
		if len(bs.corpus) == 0 {
			errors++
			continue
		}

		testImg := bs.corpus[i%len(bs.corpus)]

		detectionCount, err := bs.processImage(ctx, testImg, scenario)
		if err != nil {
			errors++
			continue
		}

		totalDetections += detectionCount
	}

	totalDuration := time.Since(startTime)

	// Capture final memory stats
	var endMem runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&endMem)

	// Calculate metrics
	metrics.TotalDuration = totalDuration
	metrics.FramesPerSecond = float64(scenario.Iterations) / totalDuration.Seconds()
	metrics.DetectionCount = totalDetections
	metrics.ErrorRate = float64(errors) / float64(scenario.Iterations)

	metrics.MemoryStats = MemoryMetrics{
		AllocBytes:      endMem.Alloc,
		TotalAllocBytes: endMem.TotalAlloc - startMem.TotalAlloc,
		SysBytes:        endMem.Sys,
		NumGC:           endMem.NumGC - startMem.NumGC,
		HeapAllocBytes:  endMem.HeapAlloc,
		HeapSysBytes:    endMem.HeapSys,
	}

	metrics.CPUStats = CPUMetrics{
		NumCPU: runtime.NumCPU(),
	}

	return metrics, nil
}

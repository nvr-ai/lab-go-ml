// Package inference - Inference sessions.
package inference

import (
	"fmt"
	"sync"
	"time"

	"github.com/nvr-ai/go-ml/inference/providers"
	ort "github.com/yalue/onnxruntime_go"
)

// Session represents a model session from the onnxruntime.
type Session struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[float32]
	Output  *ort.Tensor[float32]
}

// Close releases the resources associated with the Session.
//
// Returns:
//   - No return values.
func (s *Session) Close() {
	if s.Input != nil {
		s.Input.Destroy()
		s.Input = nil
	}
	if s.Output != nil {
		s.Output.Destroy()
		s.Output = nil
	}
	if s.Session != nil {
		s.Session.Destroy()
		s.Session = nil
	}
}

// ProfiledSession wraps an ONNX session with performance profiling capabilities
//
// This wrapper provides detailed performance metrics for optimization and debugging,
// tracking inference times, memory usage, and execution provider utilization.
type ProfiledSession struct {
	session          *ort.AdvancedSession
	config           providers.OptimizationConfig
	inferenceCount   int64
	totalTime        float64
	mu               sync.RWMutex
	profilingEnabled bool
}

// NewProfiledSession creates a new profiled ONNX session with optimization
//
// Arguments:
//   - modelPath: Path to the ONNX model file
//   - inputNames: Names of input tensors
//   - outputNames: Names of output tensors
//   - inputTensors: Input tensor objects
//   - outputTensors: Output tensor objects
//   - config: Optimization configuration
//
// Returns:
//   - *ProfiledSession: Configured profiled session
//   - error: Session creation error if any
func NewProfiledSession(
	modelPath string,
	inputNames []string,
	outputNames []string,
	inputTensors []ort.ArbitraryTensor,
	outputTensors []ort.ArbitraryTensor,
	config providers.OptimizationConfig,
) (*ProfiledSession, error) {
	options, err := providers.OptimizedSessionOptions(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create optimized session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		inputTensors,
		outputTensors,
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ProfiledSession{
		session:          session,
		config:           config,
		profilingEnabled: config.UseProfilingOptions,
	}, nil
}

// Run executes the model with performance tracking
//
// Returns:
//   - error: Execution error if any
func (ps *ProfiledSession) Run() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	start := time.Now()

	err := ps.session.Run()

	duration := float64(time.Since(start).Nanoseconds()) / 1e6 // Convert to milliseconds

	ps.inferenceCount++
	ps.totalTime += duration

	return err
}

// GetPerformanceMetrics returns comprehensive performance statistics
//
// Returns:
//   - map[string]interface{}: Performance metrics and statistics
func (ps *ProfiledSession) GetPerformanceMetrics() map[string]interface{} {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	metrics := map[string]interface{}{
		"inference_count":    ps.inferenceCount,
		"total_time_ms":      ps.totalTime,
		"profiling_enabled":  ps.profilingEnabled,
		"optimization_level": ps.config.GraphOptimizationLevel,
	}

	if ps.inferenceCount > 0 {
		metrics["average_time_ms"] = ps.totalTime / float64(ps.inferenceCount)
		metrics["throughput_fps"] = 1000.0 / (ps.totalTime / float64(ps.inferenceCount))
	}

	return metrics
}

// Destroy releases all session resources
func (ps *ProfiledSession) Destroy() {
	if ps.session != nil {
		ps.session.Destroy()
		ps.session = nil
	}
}

// ResetMetrics clears all performance counters
func (ps *ProfiledSession) ResetMetrics() {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	ps.inferenceCount = 0
	ps.totalTime = 0.0
}

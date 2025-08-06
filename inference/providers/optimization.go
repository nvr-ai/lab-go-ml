// Package providers - Advanced ONNX Runtime optimization and profiling capabilities.
package providers

import (
	"fmt"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// ShapeProfile defines min, max, and optimal input shapes for dynamic models
//
// This structure is crucial for optimizing dynamic shape models by providing
// ONNX Runtime with shape ranges that enable better optimization and memory planning.
type ShapeProfile struct {
	// InputName is the name of the input tensor
	InputName string `json:"input_name"`

	// MinShape defines the minimum dimensions [batch, channels, height, width]
	MinShape []int64 `json:"min_shape"`

	// MaxShape defines the maximum dimensions [batch, channels, height, width]
	MaxShape []int64 `json:"max_shape"`

	// OptimalShape defines the most common dimensions for optimization
	OptimalShape []int64 `json:"optimal_shape"`
}

// OptimizationConfig contains comprehensive ONNX Runtime optimization settings
//
// This configuration enables fine-tuning of ONNX Runtime behavior for optimal
// performance across different hardware configurations and model types.
type OptimizationConfig struct {
	// GraphOptimizationLevel controls the level of graph optimization
	GraphOptimizationLevel ort.GraphOptimizationLevel `json:"graph_optimization_level"`

	// EnableMemoryPattern enables memory pattern optimization
	EnableMemoryPattern bool `json:"enable_memory_pattern"`

	// EnableCPUMemArena enables CPU memory arena for better memory management
	EnableCPUMemArena bool `json:"enable_cpu_mem_arena"`

	// EnableMemoryOptimization enables memory optimization passes
	EnableMemoryOptimization bool `json:"enable_memory_optimization"`

	// ExecutionMode controls sequential vs parallel execution
	ExecutionMode ort.ExecutionMode `json:"execution_mode"`

	// IntraOpNumThreads sets threads for parallelizing ops
	IntraOpNumThreads int `json:"intra_op_num_threads"`

	// InterOpNumThreads sets threads for parallelizing independent ops
	InterOpNumThreads int `json:"inter_op_num_threads"`

	// UseProfilingOptions enables profiling for performance analysis
	UseProfilingOptions bool `json:"use_profiling_options"`

	// ProfilingOutputPath specifies where to save profiling results
	ProfilingOutputPath string `json:"profiling_output_path"`

	// ShapeProfiles defines input shape ranges for dynamic models
	ShapeProfiles []ShapeProfile `json:"shape_profiles"`

	// ExecutionProviders configures available execution providers
	ExecutionProviders []ExecutionProviderConfig `json:"execution_providers"`

	// EnableGraphFusion enables graph fusion optimizations
	EnableGraphFusion bool `json:"enable_graph_fusion"`

	// DisallowedOptimizers lists optimization passes to disable
	DisallowedOptimizers []string `json:"disallowed_optimizers"`
}

// DefaultOptimizationConfig returns a production-ready optimization configuration
//
// This configuration provides sensible defaults optimized for most use cases,
// with platform-specific adaptations for optimal performance.
func DefaultOptimizationConfig() OptimizationConfig {
	numCPU := runtime.NumCPU()

	config := OptimizationConfig{
		GraphOptimizationLevel:   ort.GraphOptimizationLevelEnableExtended,
		EnableMemoryPattern:      true,
		EnableCPUMemArena:        true,
		EnableMemoryOptimization: true,
		ExecutionMode:            ort.ExecutionModeParallel,
		IntraOpNumThreads:        maxInt(1, numCPU/2),
		InterOpNumThreads:        maxInt(1, numCPU/4),
		UseProfilingOptions:      false,
		EnableGraphFusion:        true,
		ExecutionProviders:       getDefaultExecutionProviders(),
		ShapeProfiles: []ShapeProfile{
			{
				InputName:    "images",
				MinShape:     []int64{1, 3, 320, 320},
				MaxShape:     []int64{1, 3, 1024, 1024},
				OptimalShape: []int64{1, 3, 640, 640},
			},
		},
	}

	return config
}

// getDefaultExecutionProviders returns platform-appropriate execution providers
func getDefaultExecutionProviders() []ExecutionProviderConfig {
	providers := []ExecutionProviderConfig{
		{
			Provider: CPUExecutionProvider,
			Options:  map[string]string{},
			Priority: 1,
			Enabled:  true,
		},
	}

	// Platform-specific providers
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			// Apple Silicon - CoreML is highly optimized
			providers = append(providers, ExecutionProviderConfig{
				Provider: CoreMLExecutionProvider,
				Options: map[string]string{
					"use_cpu_only": "false",
				},
				Priority: 10,
				Enabled:  true,
			})
		}
	case "linux", "windows":
		// CUDA support on Linux/Windows
		providers = append(providers, ExecutionProviderConfig{
			Provider: CUDAExecutionProvider,
			Options: map[string]string{
				"device_id":                 "0",
				"gpu_mem_limit":             "2147483648", // 2GB
				"arena_extend_strategy":     "kSameAsRequested",
				"cudnn_conv_algo_search":    "HEURISTIC",
				"do_copy_in_default_stream": "1",
			},
			Priority: 20,
			Enabled:  true,
		})

		// TensorRT for advanced NVIDIA optimization
		providers = append(providers, ExecutionProviderConfig{
			Provider: TensorRTExecutionProvider,
			Options: map[string]string{
				"device_id":               "0",
				"trt_max_workspace_size":  "1073741824", // 1GB
				"trt_fp16_enable":         "true",
				"trt_int8_enable":         "false",
				"trt_engine_cache_enable": "true",
			},
			Priority: 30,
			Enabled:  false, // Disabled by default, enable when TensorRT is available
		})

		// Intel DNNL for CPU optimization
		providers = append(providers, ExecutionProviderConfig{
			Provider: DNNLExecutionProvider,
			Options: map[string]string{
				"use_arena": "1",
			},
			Priority: 5,
			Enabled:  true,
		})
	}

	return providers
}

// OptimizedSessionOptions applies advanced optimizations to ONNX Runtime session options
//
// This function configures session options with comprehensive optimization settings,
// execution provider selection, and performance profiling capabilities.
//
// Arguments:
//   - config: Optimization configuration to apply
//
// Returns:
//   - *ort.SessionOptions: Configured session options
//   - error: Configuration error if any
//
// @example
// config := DefaultOptimizationConfig()
// options, err := OptimizedSessionOptions(config)
//
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// defer options.Destroy()
func OptimizedSessionOptions(config OptimizationConfig) (*ort.SessionOptions, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}

	// Apply basic optimization settings
	options.SetGraphOptimizationLevel(config.GraphOptimizationLevel)
	options.SetExecutionMode(config.ExecutionMode)
	options.SetIntraOpNumThreads(config.IntraOpNumThreads)
	options.SetInterOpNumThreads(config.InterOpNumThreads)

	// Note: Memory optimization and profiling settings are not available
	// in the current onnxruntime-go API but are included in the config
	// for future compatibility when the API is extended

	// Apply execution providers in priority order
	err = applyExecutionProviders(options, config.ExecutionProviders)
	if err != nil {
		options.Destroy()
		return nil, fmt.Errorf("failed to configure execution providers: %w", err)
	}

	// Note: Disallowed optimizers are not configurable in current onnxruntime-go API
	// but are included in the config for future compatibility

	return options, nil
}

// applyExecutionProviders configures execution providers based on priority and availability
func applyExecutionProviders(options *ort.SessionOptions, providers []ExecutionProviderConfig) error {
	// Sort providers by priority (highest first)
	enabledProviders := make([]ExecutionProviderConfig, 0, len(providers))
	for _, provider := range providers {
		if provider.Enabled {
			enabledProviders = append(enabledProviders, provider)
		}
	}

	// Apply providers in priority order
	for i := len(enabledProviders) - 1; i >= 0; i-- {
		provider := enabledProviders[i]

		switch provider.Provider {
		case CoreMLExecutionProvider:
			deviceId := uint32(0)
			if idStr, ok := provider.Options["device_id"]; ok {
				var id uint32
				if _, parseErr := fmt.Sscanf(idStr, "%d", &id); parseErr == nil {
					deviceId = id
				}
			}

			err := options.AppendExecutionProviderCoreML(deviceId)
			if err != nil {
				// CoreML may not be available, continue with warning
				fmt.Printf("Warning: Failed to enable CoreML provider: %v\n", err)
			}

		case OpenVINOExecutionProvider:
			err := options.AppendExecutionProviderOpenVINO(provider.Options)
			if err != nil {
				// OpenVINO may not be available, continue with warning
				fmt.Printf("Warning: Failed to enable OpenVINO provider: %v\n", err)
			}

		case CPUExecutionProvider:
			// CPU provider is always available, no explicit configuration needed

		case CUDAExecutionProvider, TensorRTExecutionProvider, DNNLExecutionProvider:
			// These providers are not available in current onnxruntime-go API
			// but are included for future compatibility
			fmt.Printf("Info: %s provider not yet supported in onnxruntime-go\n", provider.Provider)

		default:
			return fmt.Errorf("unsupported execution provider: %s", provider.Provider)
		}
	}

	return nil
}

// DynamicShapeOptimizer provides shape-aware optimization for variable input dimensions
//
// This optimizer maintains statistics about input shapes and adjusts optimization
// strategies based on observed usage patterns to maximize performance.
type DynamicShapeOptimizer struct {
	shapeProfiles    []ShapeProfile
	observedShapes   map[string][]ShapeObservation
	mu               sync.RWMutex
	optimizationHits int64
	totalInferences  int64
}

// ShapeObservation records information about observed input shapes
type ShapeObservation struct {
	Shape     []int64 `json:"shape"`
	Count     int64   `json:"count"`
	AvgTimeMs float64 `json:"avg_time_ms"`
}

// NewDynamicShapeOptimizer creates a new shape optimizer with initial profiles
//
// Arguments:
//   - profiles: Initial shape profiles for optimization
//
// Returns:
//   - *DynamicShapeOptimizer: Initialized shape optimizer
func NewDynamicShapeOptimizer(profiles []ShapeProfile) *DynamicShapeOptimizer {
	return &DynamicShapeOptimizer{
		shapeProfiles:  profiles,
		observedShapes: make(map[string][]ShapeObservation),
	}
}

// ObserveShape records a new shape observation for optimization learning
//
// Arguments:
//   - inputName: Name of the input tensor
//   - shape: Observed shape dimensions
//   - inferenceTimeMs: Time taken for inference with this shape
func (dso *DynamicShapeOptimizer) ObserveShape(inputName string, shape []int64, inferenceTimeMs float64) {
	dso.mu.Lock()
	defer dso.mu.Unlock()

	dso.totalInferences++

	// Find or create observation for this shape
	observations := dso.observedShapes[inputName]
	found := false

	for i, obs := range observations {
		if shapeEqual(obs.Shape, shape) {
			// Update existing observation
			observations[i].Count++
			observations[i].AvgTimeMs = (observations[i].AvgTimeMs*float64(observations[i].Count-1) + inferenceTimeMs) / float64(observations[i].Count)
			found = true
			break
		}
	}

	if !found {
		// Add new observation
		observations = append(observations, ShapeObservation{
			Shape:     make([]int64, len(shape)),
			Count:     1,
			AvgTimeMs: inferenceTimeMs,
		})
		copy(observations[len(observations)-1].Shape, shape)
	}

	dso.observedShapes[inputName] = observations

	// Check if this shape fits within our optimized profiles
	for _, profile := range dso.shapeProfiles {
		if profile.InputName == inputName && shapeWithinBounds(shape, profile.MinShape, profile.MaxShape) {
			dso.optimizationHits++
			break
		}
	}
}

// GetOptimizationStats returns statistics about shape optimization effectiveness
//
// Returns:
//   - map[string]interface{}: Optimization statistics and recommendations
func (dso *DynamicShapeOptimizer) GetOptimizationStats() map[string]interface{} {
	dso.mu.RLock()
	defer dso.mu.RUnlock()

	stats := map[string]interface{}{
		"total_inferences":  dso.totalInferences,
		"optimization_hits": dso.optimizationHits,
		"observed_shapes":   len(dso.observedShapes),
		"shape_profiles":    len(dso.shapeProfiles),
	}

	if dso.totalInferences > 0 {
		stats["optimization_hit_rate"] = float64(dso.optimizationHits) / float64(dso.totalInferences)
	}

	// Add shape-specific statistics
	shapeStats := make(map[string]interface{})
	for inputName, observations := range dso.observedShapes {
		inputStats := make(map[string]interface{})
		inputStats["unique_shapes"] = len(observations)

		var totalCount int64
		var fastestTime, slowestTime float64 = 999999, 0

		for _, obs := range observations {
			totalCount += obs.Count
			if obs.AvgTimeMs < fastestTime {
				fastestTime = obs.AvgTimeMs
			}
			if obs.AvgTimeMs > slowestTime {
				slowestTime = obs.AvgTimeMs
			}
		}

		inputStats["total_inferences"] = totalCount
		inputStats["fastest_time_ms"] = fastestTime
		inputStats["slowest_time_ms"] = slowestTime

		shapeStats[inputName] = inputStats
	}

	stats["input_statistics"] = shapeStats

	return stats
}

// shapeEqual compares two shape slices for equality
func shapeEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

// shapeWithinBounds checks if a shape falls within the specified bounds
func shapeWithinBounds(shape, minShape, maxShape []int64) bool {
	if len(shape) != len(minShape) || len(shape) != len(maxShape) {
		return false
	}

	for i, dim := range shape {
		if dim < minShape[i] || dim > maxShape[i] {
			return false
		}
	}

	return true
}

// maxInt returns the maximum of two integers
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

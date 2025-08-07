// Package providers - Enhanced configuration for optimized ONNX inference
package providers

import (
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"
)

// Config represents comprehensive configuration for ONNX detector with advanced optimization
//
// This configuration structure supports both backward compatibility and advanced
// optimization features including execution provider selection, shape profiling,
// and performance monitoring.
type Config struct {
	// ProviderBackend specifies the backend to use
	ProviderBackend ProviderBackend `json:"backend" yaml:"backend"`

	// ModelPath specifies the path to the ONNX model file
	ModelPath string `json:"model_path"`

	// Advanced optimization configuration
	OptimizationConfig *OptimizationConfig `json:"optimization_config,omitempty"`

	// EnableDynamicShapes allows variable input dimensions during inference
	EnableDynamicShapes bool `json:"enable_dynamic_shapes"`

	// MinInputShape defines minimum allowed input dimensions for dynamic shapes
	MinInputShape image.Point `json:"min_input_shape"`

	// MaxInputShape defines maximum allowed input dimensions for dynamic shapes
	MaxInputShape image.Point `json:"max_input_shape"`

	// EnableProfiling activates performance profiling and metrics collection
	EnableProfiling bool `json:"enable_profiling"`

	// ProfilingOutputPath specifies where to save profiling results
	ProfilingOutputPath string `json:"profiling_output_path"`

	// WarmupIterations defines how many inference runs to perform during initialization
	WarmupIterations int `json:"warmup_iterations"`

	// EnableSessionPooling allows reuse of ONNX sessions for better performance
	EnableSessionPooling bool `json:"enable_session_pooling"`

	// SessionPoolSize defines maximum number of sessions to maintain in pool
	SessionPoolSize int `json:"session_pool_size"`
}

// ExecutionProviderConfig contains configuration for specific execution providers
type ExecutionProviderConfig struct {
	// Provider specifies which execution provider to use
	Provider ProviderBackend `json:"provider"`

	// Options contains provider-specific configuration options
	Options map[string]string `json:"options"`

	// Priority determines the order in which providers are tried (higher = first)
	Priority int `json:"priority"`

	// Enabled toggles whether this provider should be used
	Enabled bool `json:"enabled"`
}

// NewProvider creates a new provider based on the configuration
func NewProvider(config Config) (*Provider[any], error) {
	switch config.ProviderBackend {
	case CPUProviderBackend:
		return NewCPUProvider(), nil
	default:
		return nil, fmt.Errorf("no matching provider backend registered: %s", config.ProviderBackend)
	}
}

// DefaultConfig returns a production-ready configuration with sensible defaults
//
// This configuration is optimized for typical object detection workloads with
// balanced performance and resource usage characteristics.
//
// Returns:
//   - Config: Production-ready configuration
//
// @example
// config := DefaultConfig()
// config.ModelPath = "path/to/model.onnx"
// session, err := NewSession(config)
func DefaultConfig() Config {
	optimizationConfig := DefaultOptimizationConfig()

	return Config{
		ModelPath:            "",
		OptimizationConfig:   &optimizationConfig,
		EnableDynamicShapes:  true,
		MinInputShape:        image.Point{X: 320, Y: 320},
		MaxInputShape:        image.Point{X: 1024, Y: 1024},
		EnableProfiling:      false,
		ProfilingOutputPath:  "",
		WarmupIterations:     3,
		EnableSessionPooling: false,
		SessionPoolSize:      4,
	}
}

// DevelopmentConfig returns a configuration optimized for development and debugging
//
// This configuration enables extensive profiling, debugging features, and
// detailed performance monitoring at the cost of some runtime performance.
//
// Returns:
//   - Config: Development-optimized configuration
func DevelopmentConfig() Config {
	config := DefaultConfig()

	// Enable comprehensive profiling and debugging
	config.EnableProfiling = true
	config.ProfilingOutputPath = "./profiling_results"
	config.WarmupIterations = 5

	// Use more conservative optimization for easier debugging
	if config.OptimizationConfig != nil {
		config.OptimizationConfig.UseProfilingOptions = true
		config.OptimizationConfig.ProfilingOutputPath = config.ProfilingOutputPath
		config.OptimizationConfig.GraphOptimizationLevel = ort.GraphOptimizationLevelDisableAll
	}

	return config
}

// HighPerformanceConfig returns a configuration optimized for maximum throughput
//
// This configuration prioritizes inference speed and throughput over memory usage
// and initialization time, suitable for high-load production environments.
//
// Returns:
//   - Config: High-performance optimized configuration
func HighPerformanceConfig() Config {
	config := DefaultConfig()

	// Enable session pooling for high throughput
	config.EnableSessionPooling = true
	config.SessionPoolSize = 8
	config.WarmupIterations = 10

	// Optimize for maximum performance
	if config.OptimizationConfig != nil {
		config.OptimizationConfig.GraphOptimizationLevel = ort.GraphOptimizationLevelEnableExtended
		config.OptimizationConfig.ExecutionMode = ort.ExecutionModeParallel
		config.OptimizationConfig.EnableMemoryOptimization = true
		config.OptimizationConfig.EnableGraphFusion = true

		// Enable high-priority execution providers
		for i := range config.OptimizationConfig.ExecutionProviders {
			provider := &config.OptimizationConfig.ExecutionProviders[i]
			switch provider.Provider {
			case CUDAExecutionProvider, TensorRTExecutionProvider:
				provider.Enabled = true
			case CoreMLExecutionProvider:
				provider.Enabled = true
			case DNNLExecutionProvider:
				provider.Enabled = true
			}
		}
	}

	return config
}

// LowLatencyConfig returns a configuration optimized for minimal inference latency
//
// This configuration prioritizes low latency over throughput, suitable for
// real-time applications where response time is critical.
//
// Returns:
//   - Config: Low-latency optimized configuration
func LowLatencyConfig() Config {
	config := DefaultConfig()

	// Optimize for low latency
	config.WarmupIterations = 15 // More warmup for consistent performance
	config.EnableSessionPooling = true
	config.SessionPoolSize = 2 // Smaller pool for lower memory usage

	if config.OptimizationConfig != nil {
		// Sequential execution for predictable timing
		config.OptimizationConfig.ExecutionMode = ort.ExecutionModeSequential
		config.OptimizationConfig.IntraOpNumThreads = 1
		config.OptimizationConfig.InterOpNumThreads = 1

		// Disable memory pattern optimization for consistent timing
		config.OptimizationConfig.EnableMemoryPattern = false
	}

	return config
}

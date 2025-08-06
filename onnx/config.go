// Package onnx - Enhanced configuration for optimized ONNX inference
package onnx

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
	// ModelPath specifies the path to the ONNX model file
	ModelPath string `json:"model_path"`
	
	// InputShape defines the default input dimensions (width, height)
	InputShape image.Point `json:"input_shape"`
	
	// ConfidenceThreshold filters detections below this confidence level
	ConfidenceThreshold float32 `json:"confidence_threshold"`
	
	// NMSThreshold controls Non-Maximum Suppression IoU threshold
	NMSThreshold float32 `json:"nms_threshold"`
	
	// RelevantClasses lists object classes to detect (empty = all classes)
	RelevantClasses []string `json:"relevant_classes"`
	
	// Legacy execution provider flags (maintained for backward compatibility)
	UseCoreML   bool `json:"use_coreml"`
	UseOpenVINO bool `json:"use_openvino"`
	
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
		ModelPath:           "",
		InputShape:          image.Point{X: 640, Y: 640},
		ConfidenceThreshold: 0.5,
		NMSThreshold:        0.7,
		RelevantClasses:     []string{},
		UseCoreML:           false, // Legacy flag, use OptimizationConfig instead
		UseOpenVINO:         false, // Legacy flag, use OptimizationConfig instead
		OptimizationConfig:  &optimizationConfig,
		EnableDynamicShapes: true,
		MinInputShape:       image.Point{X: 320, Y: 320},
		MaxInputShape:       image.Point{X: 1024, Y: 1024},
		EnableProfiling:     false,
		ProfilingOutputPath: "",
		WarmupIterations:    3,
		EnableSessionPooling: false,
		SessionPoolSize:     4,
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

// Validate checks the configuration for consistency and completeness
//
// This method performs comprehensive validation of all configuration parameters
// to ensure they are within valid ranges and logically consistent.
//
// Returns:
//   - error: Validation error if configuration is invalid, nil otherwise
//
// @example
// config := DefaultConfig()
// config.ModelPath = "model.onnx"
// if err := config.Validate(); err != nil {
//     log.Fatal("Invalid configuration:", err)
// }
func (c *Config) Validate() error {
	// Required fields validation
	if c.ModelPath == "" {
		return fmt.Errorf("ModelPath is required")
	}
	
	if c.InputShape.X <= 0 || c.InputShape.Y <= 0 {
		return fmt.Errorf("InputShape must have positive dimensions, got %v", c.InputShape)
	}
	
	// Threshold validation
	if c.ConfidenceThreshold < 0 || c.ConfidenceThreshold > 1 {
		return fmt.Errorf("ConfidenceThreshold must be between 0 and 1, got %f", c.ConfidenceThreshold)
	}
	
	if c.NMSThreshold < 0 || c.NMSThreshold > 1 {
		return fmt.Errorf("NMSThreshold must be between 0 and 1, got %f", c.NMSThreshold)
	}
	
	// Dynamic shape validation
	if c.EnableDynamicShapes {
		if c.MinInputShape.X <= 0 || c.MinInputShape.Y <= 0 {
			return fmt.Errorf("MinInputShape must have positive dimensions when dynamic shapes are enabled")
		}
		
		if c.MaxInputShape.X <= 0 || c.MaxInputShape.Y <= 0 {
			return fmt.Errorf("MaxInputShape must have positive dimensions when dynamic shapes are enabled")
		}
		
		if c.MinInputShape.X > c.MaxInputShape.X || c.MinInputShape.Y > c.MaxInputShape.Y {
			return fmt.Errorf("MinInputShape must be smaller than MaxInputShape")
		}
		
		if c.InputShape.X < c.MinInputShape.X || c.InputShape.X > c.MaxInputShape.X ||
		   c.InputShape.Y < c.MinInputShape.Y || c.InputShape.Y > c.MaxInputShape.Y {
			return fmt.Errorf("InputShape must be within MinInputShape and MaxInputShape bounds")
		}
	}
	
	// Performance settings validation
	if c.WarmupIterations < 0 {
		return fmt.Errorf("WarmupIterations must be non-negative, got %d", c.WarmupIterations)
	}
	
	if c.EnableSessionPooling && c.SessionPoolSize <= 0 {
		return fmt.Errorf("SessionPoolSize must be positive when session pooling is enabled")
	}
	
	return nil
}

// GetExecutionProviders returns the list of enabled execution providers in priority order
//
// This method extracts execution provider configuration from both legacy flags
// and modern OptimizationConfig for backward compatibility.
//
// Returns:
//   - []ExecutionProviderConfig: Enabled execution providers in priority order
func (c *Config) GetExecutionProviders() []ExecutionProviderConfig {
	var providers []ExecutionProviderConfig
	
	// Handle legacy configuration flags
	if c.UseCoreML {
		providers = append(providers, ExecutionProviderConfig{
			Provider: CoreMLExecutionProvider,
			Options:  map[string]string{},
			Priority: 10,
			Enabled:  true,
		})
	}
	
	if c.UseOpenVINO {
		providers = append(providers, ExecutionProviderConfig{
			Provider: OpenVINOExecutionProvider,
			Options:  map[string]string{},
			Priority: 8,
			Enabled:  true,
		})
	}
	
	// Add providers from OptimizationConfig
	if c.OptimizationConfig != nil {
		for _, provider := range c.OptimizationConfig.ExecutionProviders {
			if provider.Enabled {
				providers = append(providers, provider)
			}
		}
	}
	
	// Add default CPU provider if no others specified
	if len(providers) == 0 {
		providers = append(providers, ExecutionProviderConfig{
			Provider: CPUExecutionProvider,
			Options:  map[string]string{},
			Priority: 1,
			Enabled:  true,
		})
	}
	
	return providers
}

// Clone creates a deep copy of the configuration
//
// Returns:
//   - *Config: Deep copy of the configuration
func (c *Config) Clone() *Config {
	clone := &Config{
		ModelPath:            c.ModelPath,
		InputShape:           c.InputShape,
		ConfidenceThreshold:  c.ConfidenceThreshold,
		NMSThreshold:         c.NMSThreshold,
		RelevantClasses:      make([]string, len(c.RelevantClasses)),
		UseCoreML:            c.UseCoreML,
		UseOpenVINO:          c.UseOpenVINO,
		EnableDynamicShapes:  c.EnableDynamicShapes,
		MinInputShape:        c.MinInputShape,
		MaxInputShape:        c.MaxInputShape,
		EnableProfiling:      c.EnableProfiling,
		ProfilingOutputPath:  c.ProfilingOutputPath,
		WarmupIterations:     c.WarmupIterations,
		EnableSessionPooling: c.EnableSessionPooling,
		SessionPoolSize:      c.SessionPoolSize,
	}
	
	copy(clone.RelevantClasses, c.RelevantClasses)
	
	// Deep copy optimization config if present
	if c.OptimizationConfig != nil {
		optimConfig := *c.OptimizationConfig
		clone.OptimizationConfig = &optimConfig
		
		// Deep copy slices within optimization config
		if len(optimConfig.ShapeProfiles) > 0 {
			clone.OptimizationConfig.ShapeProfiles = make([]ShapeProfile, len(optimConfig.ShapeProfiles))
			copy(clone.OptimizationConfig.ShapeProfiles, optimConfig.ShapeProfiles)
		}
		
		if len(optimConfig.ExecutionProviders) > 0 {
			clone.OptimizationConfig.ExecutionProviders = make([]ExecutionProviderConfig, len(optimConfig.ExecutionProviders))
			copy(clone.OptimizationConfig.ExecutionProviders, optimConfig.ExecutionProviders)
		}
		
		if len(optimConfig.DisallowedOptimizers) > 0 {
			clone.OptimizationConfig.DisallowedOptimizers = make([]string, len(optimConfig.DisallowedOptimizers))
			copy(clone.OptimizationConfig.DisallowedOptimizers, optimConfig.DisallowedOptimizers)
		}
	}
	
	return clone
}

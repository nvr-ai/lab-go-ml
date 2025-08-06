// Package onnx - Comprehensive testing for ONNX Runtime optimization features
package onnx

import (
	"image"
	"runtime"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDefaultOptimizationConfig tests the default optimization configuration
func TestDefaultOptimizationConfig(t *testing.T) {
	config := DefaultOptimizationConfig()

	// Verify basic settings
	assert.Equal(t, ort.GraphOptimizationLevelEnableExtended, config.GraphOptimizationLevel)
	assert.True(t, config.EnableMemoryPattern)
	assert.True(t, config.EnableCPUMemArena)
	assert.True(t, config.EnableMemoryOptimization)
	assert.Equal(t, ort.ExecutionModeParallel, config.ExecutionMode)
	assert.True(t, config.EnableGraphFusion)

	// Verify thread configuration is reasonable
	numCPU := runtime.NumCPU()
	expectedIntraOp := max(1, numCPU/2)
	expectedInterOp := max(1, numCPU/4)
	
	assert.Equal(t, expectedIntraOp, config.IntraOpNumThreads)
	assert.Equal(t, expectedInterOp, config.InterOpNumThreads)

	// Verify shape profiles
	assert.NotEmpty(t, config.ShapeProfiles)
	profile := config.ShapeProfiles[0]
	assert.Equal(t, "images", profile.InputName)
	assert.Equal(t, []int64{1, 3, 320, 320}, profile.MinShape)
	assert.Equal(t, []int64{1, 3, 1024, 1024}, profile.MaxShape)
	assert.Equal(t, []int64{1, 3, 640, 640}, profile.OptimalShape)

	// Verify execution providers
	assert.NotEmpty(t, config.ExecutionProviders)
	
	// Should always have CPU provider
	hasCPU := false
	for _, provider := range config.ExecutionProviders {
		if provider.Provider == CPUExecutionProvider && provider.Enabled {
			hasCPU = true
			break
		}
	}
	assert.True(t, hasCPU, "CPU execution provider should be enabled by default")
}

// TestPlatformSpecificProviders tests platform-specific execution providers
func TestPlatformSpecificProviders(t *testing.T) {
	providers := getDefaultExecutionProviders()
	
	// Should always have CPU provider
	hasCPU := false
	for _, provider := range providers {
		if provider.Provider == CPUExecutionProvider {
			hasCPU = true
			assert.True(t, provider.Enabled)
			assert.Equal(t, 1, provider.Priority)
			break
		}
	}
	assert.True(t, hasCPU, "CPU provider should always be available")

	// Platform-specific checks
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			// Apple Silicon should have CoreML
			hasCoreML := false
			for _, provider := range providers {
				if provider.Provider == CoreMLExecutionProvider {
					hasCoreML = true
					assert.True(t, provider.Enabled)
					assert.Greater(t, provider.Priority, 1)
					break
				}
			}
			assert.True(t, hasCoreML, "Apple Silicon should have CoreML provider")
		}
	case "linux", "windows":
		// Should have CUDA and DNNL options
		hasCUDA := false
		hasDNNL := false
		
		for _, provider := range providers {
			if provider.Provider == CUDAExecutionProvider {
				hasCUDA = true
				assert.True(t, provider.Enabled)
				assert.Greater(t, provider.Priority, 1)
			}
			if provider.Provider == DNNLExecutionProvider {
				hasDNNL = true
				assert.True(t, provider.Enabled)
				assert.Greater(t, provider.Priority, 1)
			}
		}
		
		assert.True(t, hasCUDA, "Linux/Windows should have CUDA provider")
		assert.True(t, hasDNNL, "Linux/Windows should have DNNL provider")
	}
}

// TestOptimizedSessionOptions tests session option configuration
func TestOptimizedSessionOptions(t *testing.T) {
	config := DefaultOptimizationConfig()
	
	// Test successful configuration
	options, err := OptimizedSessionOptions(config)
	require.NoError(t, err)
	require.NotNil(t, options)
	defer options.Destroy()

	// Test with profiling enabled
	config.UseProfilingOptions = true
	config.ProfilingOutputPath = "./test_profiling"
	
	optionsWithProfiling, err := OptimizedSessionOptions(config)
	require.NoError(t, err)
	require.NotNil(t, optionsWithProfiling)
	defer optionsWithProfiling.Destroy()

	// Test with disabled optimizers
	config.DisallowedOptimizers = []string{"ConstantFolding", "EliminateIdentity"}
	
	optionsWithDisabled, err := OptimizedSessionOptions(config)
	require.NoError(t, err)
	require.NotNil(t, optionsWithDisabled)
	defer optionsWithDisabled.Destroy()
}

// TestExecutionProviderConfiguration tests execution provider setup
func TestExecutionProviderConfiguration(t *testing.T) {
	tests := []struct {
		name      string
		providers []ExecutionProviderConfig
		expectErr bool
	}{
		{
			name: "CPU only",
			providers: []ExecutionProviderConfig{
				{Provider: CPUExecutionProvider, Enabled: true, Priority: 1},
			},
			expectErr: false,
		},
		{
			name: "Multiple providers with priorities",
			providers: []ExecutionProviderConfig{
				{Provider: CPUExecutionProvider, Enabled: true, Priority: 1},
				{Provider: CoreMLExecutionProvider, Enabled: true, Priority: 10},
			},
			expectErr: false,
		},
		{
			name: "Disabled providers",
			providers: []ExecutionProviderConfig{
				{Provider: CPUExecutionProvider, Enabled: false, Priority: 1},
				{Provider: CUDAExecutionProvider, Enabled: false, Priority: 20},
			},
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options, err := ort.NewSessionOptions()
			require.NoError(t, err)
			defer options.Destroy()

			err = applyExecutionProviders(options, tt.providers)
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestDynamicShapeOptimizer tests shape optimization functionality
func TestDynamicShapeOptimizer(t *testing.T) {
	profiles := []ShapeProfile{
		{
			InputName:    "input",
			MinShape:     []int64{1, 3, 320, 320},
			MaxShape:     []int64{1, 3, 640, 640},
			OptimalShape: []int64{1, 3, 480, 480},
		},
	}

	optimizer := NewDynamicShapeOptimizer(profiles)
	
	// Test initial state
	stats := optimizer.GetOptimizationStats()
	assert.Equal(t, int64(0), stats["total_inferences"])
	assert.Equal(t, int64(0), stats["optimization_hits"])
	
	// Test shape observations
	optimizer.ObserveShape("input", []int64{1, 3, 480, 480}, 15.5) // Within bounds
	optimizer.ObserveShape("input", []int64{1, 3, 320, 320}, 12.0) // Within bounds
	optimizer.ObserveShape("input", []int64{1, 3, 800, 800}, 25.0) // Outside bounds
	optimizer.ObserveShape("input", []int64{1, 3, 480, 480}, 16.0) // Duplicate shape

	stats = optimizer.GetOptimizationStats()
	assert.Equal(t, int64(4), stats["total_inferences"])
	assert.Equal(t, int64(3), stats["optimization_hits"]) // First 3 are within bounds
	
	hitRate, ok := stats["optimization_hit_rate"].(float64)
	assert.True(t, ok)
	assert.InDelta(t, 0.75, hitRate, 0.01) // 3/4 = 0.75

	// Verify input statistics
	inputStats, ok := stats["input_statistics"].(map[string]interface{})
	assert.True(t, ok)
	
	inputInfo, ok := inputStats["input"].(map[string]interface{})
	assert.True(t, ok)
	
	uniqueShapes, ok := inputInfo["unique_shapes"].(int)
	assert.True(t, ok)
	assert.Equal(t, 3, uniqueShapes)
}

// TestShapeProfileValidation tests shape profile validation functions
func TestShapeProfileValidation(t *testing.T) {
	tests := []struct {
		name     string
		shape1   []int64
		shape2   []int64
		expected bool
	}{
		{
			name:     "Equal shapes",
			shape1:   []int64{1, 3, 640, 640},
			shape2:   []int64{1, 3, 640, 640},
			expected: true,
		},
		{
			name:     "Different shapes",
			shape1:   []int64{1, 3, 640, 640},
			shape2:   []int64{1, 3, 480, 480},
			expected: false,
		},
		{
			name:     "Different lengths",
			shape1:   []int64{1, 3, 640},
			shape2:   []int64{1, 3, 640, 640},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shapeEqual(tt.shape1, tt.shape2)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestShapeWithinBounds tests shape bounds checking
func TestShapeWithinBounds(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int64
		minShape []int64
		maxShape []int64
		expected bool
	}{
		{
			name:     "Shape within bounds",
			shape:    []int64{1, 3, 480, 480},
			minShape: []int64{1, 3, 320, 320},
			maxShape: []int64{1, 3, 640, 640},
			expected: true,
		},
		{
			name:     "Shape at minimum bound",
			shape:    []int64{1, 3, 320, 320},
			minShape: []int64{1, 3, 320, 320},
			maxShape: []int64{1, 3, 640, 640},
			expected: true,
		},
		{
			name:     "Shape at maximum bound",
			shape:    []int64{1, 3, 640, 640},
			minShape: []int64{1, 3, 320, 320},
			maxShape: []int64{1, 3, 640, 640},
			expected: true,
		},
		{
			name:     "Shape below minimum",
			shape:    []int64{1, 3, 200, 200},
			minShape: []int64{1, 3, 320, 320},
			maxShape: []int64{1, 3, 640, 640},
			expected: false,
		},
		{
			name:     "Shape above maximum",
			shape:    []int64{1, 3, 800, 800},
			minShape: []int64{1, 3, 320, 320},
			maxShape: []int64{1, 3, 640, 640},
			expected: false,
		},
		{
			name:     "Different dimensions",
			shape:    []int64{1, 3, 480},
			minShape: []int64{1, 3, 320, 320},
			maxShape: []int64{1, 3, 640, 640},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shapeWithinBounds(tt.shape, tt.minShape, tt.maxShape)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestConfigValidation tests configuration validation
func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      Config
		expectError bool
		errorMsg    string
	}{
		{
			name:        "Valid default config",
			config:      DefaultConfig(),
			expectError: true, // ModelPath is empty
			errorMsg:    "ModelPath is required",
		},
		{
			name: "Valid complete config",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				return c
			}(),
			expectError: false,
		},
		{
			name: "Invalid confidence threshold",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				c.ConfidenceThreshold = -0.1
				return c
			}(),
			expectError: true,
			errorMsg:    "ConfidenceThreshold must be between 0 and 1",
		},
		{
			name: "Invalid NMS threshold",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				c.NMSThreshold = 1.5
				return c
			}(),
			expectError: true,
			errorMsg:    "NMSThreshold must be between 0 and 1",
		},
		{
			name: "Invalid input shape",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				c.InputShape = image.Point{X: -100, Y: 100}
				return c
			}(),
			expectError: true,
			errorMsg:    "InputShape must have positive dimensions",
		},
		{
			name: "Invalid dynamic shape bounds",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				c.EnableDynamicShapes = true
				c.MinInputShape = image.Point{X: 640, Y: 640}
				c.MaxInputShape = image.Point{X: 320, Y: 320}
				return c
			}(),
			expectError: true,
			errorMsg:    "MinInputShape must be smaller than MaxInputShape",
		},
		{
			name: "Input shape outside dynamic bounds",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				c.EnableDynamicShapes = true
				c.InputShape = image.Point{X: 200, Y: 200}
				c.MinInputShape = image.Point{X: 320, Y: 320}
				c.MaxInputShape = image.Point{X: 640, Y: 640}
				return c
			}(),
			expectError: true,
			errorMsg:    "InputShape must be within MinInputShape and MaxInputShape bounds",
		},
		{
			name: "Invalid session pool size",
			config: func() Config {
				c := DefaultConfig()
				c.ModelPath = "test.onnx"
				c.EnableSessionPooling = true
				c.SessionPoolSize = -1
				return c
			}(),
			expectError: true,
			errorMsg:    "SessionPoolSize must be positive when session pooling is enabled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			
			if tt.expectError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestConfigurationPresets tests different configuration presets
func TestConfigurationPresets(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := DefaultConfig()
		assert.NotNil(t, config.OptimizationConfig)
		assert.True(t, config.EnableDynamicShapes)
		assert.False(t, config.EnableProfiling)
		assert.Equal(t, 3, config.WarmupIterations)
	})

	t.Run("DevelopmentConfig", func(t *testing.T) {
		config := DevelopmentConfig()
		assert.True(t, config.EnableProfiling)
		assert.NotEmpty(t, config.ProfilingOutputPath)
		assert.Greater(t, config.WarmupIterations, 3)
		assert.NotNil(t, config.OptimizationConfig)
		assert.True(t, config.OptimizationConfig.UseProfilingOptions)
		assert.Equal(t, ort.GraphOptimizationLevelDisableAll, config.OptimizationConfig.GraphOptimizationLevel)
	})

	t.Run("HighPerformanceConfig", func(t *testing.T) {
		config := HighPerformanceConfig()
		assert.True(t, config.EnableSessionPooling)
		assert.Greater(t, config.SessionPoolSize, 4)
		assert.Greater(t, config.WarmupIterations, 5)
		assert.NotNil(t, config.OptimizationConfig)
		assert.Equal(t, ort.GraphOptimizationLevelEnableExtended, config.OptimizationConfig.GraphOptimizationLevel)
		assert.Equal(t, ort.ExecutionModeParallel, config.OptimizationConfig.ExecutionMode)
	})

	t.Run("LowLatencyConfig", func(t *testing.T) {
		config := LowLatencyConfig()
		assert.True(t, config.EnableSessionPooling)
		assert.LessOrEqual(t, config.SessionPoolSize, 2)
		assert.Greater(t, config.WarmupIterations, 10)
		assert.NotNil(t, config.OptimizationConfig)
		assert.Equal(t, ort.ExecutionModeSequential, config.OptimizationConfig.ExecutionMode)
		assert.Equal(t, 1, config.OptimizationConfig.IntraOpNumThreads)
		assert.Equal(t, 1, config.OptimizationConfig.InterOpNumThreads)
	})
}

// TestConfigCloning tests configuration deep copying
func TestConfigCloning(t *testing.T) {
	original := DefaultConfig()
	original.ModelPath = "original.onnx"
	original.RelevantClasses = []string{"person", "car"}
	original.OptimizationConfig.DisallowedOptimizers = []string{"optimizer1", "optimizer2"}

	cloned := original.Clone()

	// Test that values are copied
	assert.Equal(t, original.ModelPath, cloned.ModelPath)
	assert.Equal(t, original.RelevantClasses, cloned.RelevantClasses)
	assert.Equal(t, original.OptimizationConfig.DisallowedOptimizers, cloned.OptimizationConfig.DisallowedOptimizers)

	// Test that slices are independent
	original.RelevantClasses[0] = "modified"
	assert.NotEqual(t, original.RelevantClasses[0], cloned.RelevantClasses[0])

	original.OptimizationConfig.DisallowedOptimizers[0] = "modified"
	assert.NotEqual(t, original.OptimizationConfig.DisallowedOptimizers[0], cloned.OptimizationConfig.DisallowedOptimizers[0])
}

// TestExecutionProviderExtraction tests execution provider configuration extraction
func TestExecutionProviderExtraction(t *testing.T) {
	t.Run("Legacy flags only", func(t *testing.T) {
		config := Config{
			UseCoreML:   true,
			UseOpenVINO: true,
		}

		providers := config.GetExecutionProviders()
		
		hasCoreML := false
		hasOpenVINO := false
		
		for _, provider := range providers {
			if provider.Provider == CoreMLExecutionProvider && provider.Enabled {
				hasCoreML = true
			}
			if provider.Provider == OpenVINOExecutionProvider && provider.Enabled {
				hasOpenVINO = true
			}
		}
		
		assert.True(t, hasCoreML)
		assert.True(t, hasOpenVINO)
	})

	t.Run("OptimizationConfig providers", func(t *testing.T) {
		optimConfig := DefaultOptimizationConfig()
		config := Config{
			OptimizationConfig: &optimConfig,
		}

		providers := config.GetExecutionProviders()
		assert.NotEmpty(t, providers)
		
		// Should have at least CPU provider
		hasCPU := false
		for _, provider := range providers {
			if provider.Provider == CPUExecutionProvider && provider.Enabled {
				hasCPU = true
				break
			}
		}
		assert.True(t, hasCPU)
	})

	t.Run("No providers specified", func(t *testing.T) {
		config := Config{}
		
		providers := config.GetExecutionProviders()
		assert.Len(t, providers, 1)
		assert.Equal(t, CPUExecutionProvider, providers[0].Provider)
		assert.True(t, providers[0].Enabled)
	})
}

// BenchmarkOptimizationConfig benchmarks configuration creation
func BenchmarkOptimizationConfig(b *testing.B) {
	b.Run("DefaultOptimizationConfig", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = DefaultOptimizationConfig()
		}
	})

	b.Run("DefaultConfig", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = DefaultConfig()
		}
	})

	b.Run("ConfigValidation", func(b *testing.B) {
		config := DefaultConfig()
		config.ModelPath = "test.onnx"
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = config.Validate()
		}
	})

	b.Run("ConfigCloning", func(b *testing.B) {
		config := DefaultConfig()
		config.ModelPath = "test.onnx"
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = config.Clone()
		}
	})
}
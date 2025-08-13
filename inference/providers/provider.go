// Package providers - Provider interface for execution providers.
package providers

import (
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"
)

// ProviderOptions is a marker interface for provider-specific config.
type ProviderOptions interface {
	isProviderOptions()
}

// Config represents comprehensive configuration for ONNX detector with advanced optimization
//
// This configuration structure supports both backward compatibility and advanced
// optimization features including execution provider selection, shape profiling,
// and performance monitoring.
type Config struct {
	// Backend specifies the backend to use
	Backend ProviderBackend `json:"backend" yaml:"backend"`

	// Options contains provider-specific configuration options
	Options ProviderOptions `json:"options" yaml:"options"`

	// Advanced optimization configuration
	Optimization *OptimizationConfig `json:"optimization,omitempty"`

	// MinShape defines minimum allowed input dimensions for dynamic shapes
	MinShape image.Point `json:"min_shape" yaml:"min_shape"`

	// MaxShape defines maximum allowed input dimensions for dynamic shapes
	MaxShape image.Point `json:"max_shape" yaml:"max_shape"`

	// Warmup defines how many inference runs to perform during initialization
	Warmup int `json:"warmup" yaml:"warmup"`
}

// NewConfig creates a new configuration for the providers
func NewConfig(args Config) (*Config, error) {
	if args.Backend == "" {
		return nil, fmt.Errorf("backend is required")
	}

	if args.MaxShape.X < 320 {
		return nil, fmt.Errorf(
			"max_shape.x must be greater than or equal to 320, got %d",
			args.MaxShape.X,
		)
	}
	if args.MaxShape.Y < 320 {
		return nil, fmt.Errorf(
			"max_shape.y must be greater than or equal to 320, got %d",
			args.MaxShape.Y,
		)
	}
	if args.MinShape.X < 320 {
		return nil, fmt.Errorf(
			"min_shape.x must be greater than or equal to 320, got %d",
			args.MinShape.X,
		)
	}
	if args.MinShape.Y < 320 {
		return nil, fmt.Errorf(
			"min_shape.y must be greater than or equal to 320, got %d",
			args.MinShape.Y,
		)
	}

	return &Config{
		Backend:  args.Backend,
		MinShape: args.MinShape,
		MaxShape: args.MaxShape,
		Warmup:   args.Warmup,
		Optimization: &OptimizationConfig{
			GraphOptimizationLevel:   ort.GraphOptimizationLevelEnableExtended,
			ExecutionMode:            ort.ExecutionModeParallel,
			EnableMemoryOptimization: true,
			EnableGraphFusion:        true,
		},
	}, nil
}

// ProviderBackend represents different ONNX Runtime execution providers
type ProviderBackend string

// ProviderConstructor represents a function that creates a new execution provider.
type ProviderConstructor func(args interface{}) ExecutionProvider

// Provider represents the contract that all execution providers must implement.
type Provider[T ExecutionProvider] struct {
	Backend  T
	Priority int
	Enabled  bool
}

// ExecutionProvider represents the contract that all execution providers must implement.
type ExecutionProvider interface {
	Backend() ProviderBackend
	Options() ProviderOptions
	Execute(input []byte) ([]byte, error)
}

// NewProvider creates a new provider based on the required backend.
//
// Arguments:
//   - backend: The backend to use.
//   - options: The options for the provider.
//
// Returns:
//   - *Provider[any]: The new provider.
//   - error: An error if the provider creation fails.
func NewProvider(options ProviderOptions) (ExecutionProvider, error) {
	switch opts := options.(type) {
	case CoreMLOptions:
		return NewCoreMLProvider(opts), nil
	case OpenVINOOptions:
		return NewOpenVINOProvider(opts), nil
	case CUDAOptions:
		return NewCUDAProvider(opts), nil
	default:
		return nil, fmt.Errorf("unsupported provider options type: %T", opts)
	}
}

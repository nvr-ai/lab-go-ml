// Package providers - Enhanced configuration for optimized ONNX inference
package providers

// ExecutionProviderConfig contains configuration for specific execution providers
type ExecutionProviderConfig struct {
	// Provider specifies which execution provider to use
	Provider ProviderBackend `json:"provider" yaml:"provider"`

	// Options contains provider-specific configuration options
	Options map[string]string `json:"options" yaml:"options"`

	// Priority determines the order in which providers are tried (higher = first)
	Priority int `json:"priority" yaml:"priority"`
}

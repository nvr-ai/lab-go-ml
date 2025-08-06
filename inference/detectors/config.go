// Package detectors - Enhanced configuration for optimized ONNX inference
package detectors

import (
	"image"

	"github.com/nvr-ai/go-ml/inference/providers"
)

// Config represents comprehensive configuration for ONNX detector with advanced optimization
//
// This configuration structure supports both backward compatibility and advanced
// optimization features including execution provider selection, shape profiling,
// and performance monitoring.
type Config struct {
	// Backend execution provider configuration
	Provider providers.Config `json:"provider"`

	// InputShape defines the default input dimensions (width, height)
	InputShape image.Point `json:"input_shape"`

	// ConfidenceThreshold filters detections below this confidence level
	ConfidenceThreshold float32 `json:"confidence_threshold"`

	// NMSThreshold controls Non-Maximum Suppression IoU threshold
	NMSThreshold float32 `json:"nms_threshold"`

	// RelevantClasses lists object classes to detect (empty = all classes)
	RelevantClasses []string `json:"relevant_classes"`
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
	return Config{
		Provider:            providers.DefaultConfig(),
		InputShape:          image.Point{X: 640, Y: 640},
		ConfidenceThreshold: 0.5,
		NMSThreshold:        0.7,
		RelevantClasses:     []string{},
	}
}

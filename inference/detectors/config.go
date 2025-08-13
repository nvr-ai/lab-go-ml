// Package detectors - Enhanced configuration for optimized ONNX inference
package detectors

import (
	"fmt"
	"image"

	"github.com/nvr-ai/go-ml/inference/providers"
	"github.com/nvr-ai/go-ml/models"
	"github.com/nvr-ai/go-ml/models/model"
)

// Config represents comprehensive configuration for ONNX detector with advanced optimization
//
// This configuration structure supports both backward compatibility and advanced
// optimization features including execution provider selection, shape profiling,
// and performance monitoring.
type Config struct {
	// Backend execution provider configuration
	Provider providers.Config `json:"provider"`
	// Model represents the model to use for inference
	Model model.NewModelArgs `json:"model"`
	// Shape defines the default input dimensions (width, height)
	Shape image.Point `json:"shape"`
	// Confidence filters detections below this confidence level
	Confidence float32 `json:"confidence"`
	// Nms controls Non-Maximum Suppression IoU threshold
	Nms float32 `json:"nms"`
	// Classes lists object classes to detect (empty = all classes)
	Classes []models.ClassName `json:"classes"`
}

// NewConfig returns a production-ready configuration with sensible defaults
//
// This configuration is optimized for typical object detection workloads with
// balanced performance and resource usage characteristics.
//
// Returns:
//   - Config: Production-ready configuration
//   - error: An error if the configuration is not valid.
func NewConfig(args Config) (*Config, error) {
	provider, err := providers.NewConfig(args.Provider)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider: %w", err)
	}

	return &Config{
		Provider:   *provider,
		Model:      args.Model,
		Shape:      args.Shape,
		Confidence: args.Confidence,
		Nms:        args.Nms,
		Classes:    args.Classes,
	}, nil
}

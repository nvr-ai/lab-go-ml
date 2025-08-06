// Package inference - Inference engine interface and implementations.
package inference

import (
	"context"
	"image"
)

// Engine defines the interface for ML inference engines
type Engine interface {
	LoadModel(modelPath string, config map[string]interface{}) error
	Predict(ctx context.Context, img image.Image) (interface{}, error)
	Close() error
	GetModelInfo() map[string]interface{}
	WarmUp(runs int) error
}

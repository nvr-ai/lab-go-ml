// Package postprocess - Postprocessing utilities for models.
package postprocess

import "github.com/nvr-ai/go-ml/images"

// Result represents a single detection result.
type Result struct {
	// The bounding box of the result.
	Box images.Rect
	// The confidence score of the result.
	Score float32
	// The predicted class index of the result.
	Class int
}

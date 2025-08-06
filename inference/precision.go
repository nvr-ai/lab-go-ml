// Package inference - This file provides common utilities for inference tasks.
package inference

// Precision represents the precision of a model.
type Precision string

// Precision constants are the supported precisions for inference.
const (
	PrecisionINT8 Precision = "INT8"
	PrecisionFP8  Precision = "FP8"
	PrecisionFP16 Precision = "FP16"
	PrecisionFP32 Precision = "FP32"
)

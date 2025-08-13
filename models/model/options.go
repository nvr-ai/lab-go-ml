// Package model - Model options.
//
// See:
// https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options
package model

// Precision represents the precision of the model.
type Precision string

const (
	// PrecisionAccuracy represents the accuracy of the model.
	// (OpenVINO's default input precision type.)
	PrecisionAccuracy Precision = "ACCURACY"
	// PrecisionFP32 represents the speed of the model.
	PrecisionFP32 Precision = "FP32"
	// PrecisionFP16 represents 16-bit floating point precision.
	PrecisionFP16 Precision = "FP16"
	// PrecisionINT8 represents 8-bit integer precision.
	PrecisionINT8 Precision = "INT8"
)

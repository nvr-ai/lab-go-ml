package onnx

import "image"

// Config for ONNX detector
type Config struct {
	ModelPath           string
	InputShape          image.Point
	ConfidenceThreshold float32
	NMSThreshold        float32
	RelevantClasses     []string
	UseCoreML           bool
	DisableONNXRuntime  bool // New field to disable ONNX Runtime when library is unavailable
}

// Package inference - Inference engine interface and implementations
package inference

// EngineType is the type of the engine
type EngineType string

const (
	// EngineOpenVINO is the OpenVINO engine that uses the OpenVINO library
	EngineOpenVINO EngineType = "openvino"
	// EngineCoreML is the CoreML engine that uses the CoreML library
	EngineCoreML EngineType = "coreml"
)

// Engines is a list of all supported engines
var Engines = []EngineType{EngineOpenVINO, EngineCoreML}

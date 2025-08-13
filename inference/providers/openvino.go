// Package providers - CPU based execution provider.
package providers

const (
	// OpenVINOProviderBackend uses Intel OpenVINO for inference optimization.
	OpenVINOProviderBackend ProviderBackend = "openvino"
)

// OpenVINOProvider implements the ExecutionProvider interface.
type OpenVINOProvider struct {
	DeviceID     string
	DeviceType   string
	Precision    float32
	NumOfThreads int
}

// OpenVINOOptions contains arguments for the OpenVINO provider.
// See:
// https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options
type OpenVINOOptions struct {
	DeviceID string `json:"deviceID"             yaml:"deviceID"`
	// Overrides the accelerator hardware type with these values at runtime. If this option is not
	// explicitly set, default hardware specified during build is used.
	DeviceType string `json:"deviceType"           yaml:"deviceType"`
	// Supported precisions for HW {CPU:FP32, GPU:[FP32, FP16, ACCURACY], NPU:FP16}. Default precision
	// for HW for optimized performance {CPU:FP32, GPU:FP16, NPU:FP16}. To execute model with the
	// default input precision, select ACCURACY precision type.
	Precision float32 `json:"precision"            yaml:"precision"`
	// Overrides the accelerator default value of number of threads with this value at runtime.
	// If this option is not explicitly set, default value of 8 during build time will be used for
	// inference.
	NumOfThreads int `json:"numOfThreads"         yaml:"numOfThreads"`
	// Overrides the accelerator default streams with this value at runtime. If this option is not
	// explicitly set, default value of 1, performance for latency is used during build time will be
	// used for inference.
	NumStreams int `json:"numStreams"           yaml:"numStreams"`
	// This option enables rewriting dynamic shaped models to static shape at runtime and execute.
	DisableDynamicShapes bool `json:"disableDynamicShapes" yaml:"disableDynamicShapes"`
	// This option configures which models should be allocated to the best resource.
	ModelPriority int `json:"modelPriority"        yaml:"modelPriority"`
}

// isProviderOptions is a marker function to ensure the options are valid.
func (OpenVINOOptions) isProviderOptions() {}

// Backend returns the backend of the OpenVINO provider.
func (p *OpenVINOProvider) Backend() ProviderBackend {
	return OpenVINOProviderBackend
}

// Options returns the options of the OpenVINO provider.
func (p *OpenVINOProvider) Options() ProviderOptions {
	return OpenVINOOptions{
		DeviceID:     p.DeviceID,
		DeviceType:   p.DeviceType,
		Precision:    p.Precision,
		NumOfThreads: p.NumOfThreads,
	}
}

// Execute executes the OpenVINO provider.
func (p *OpenVINOProvider) Execute(input []byte) ([]byte, error) {
	// CUDA execution logic
	return input, nil
}

// NewOpenVINOProvider creates a new OpenVINO provider.
func NewOpenVINOProvider(args OpenVINOOptions) *OpenVINOProvider {
	return &OpenVINOProvider{
		DeviceID:     args.DeviceID,
		DeviceType:   args.DeviceType,
		Precision:    args.Precision,
		NumOfThreads: args.NumOfThreads,
	}
}

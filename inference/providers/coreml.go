// Package providers - CPU based execution provider.
package providers

const (
	// CoreMLProviderBackend uses Apple CoreML for macOS/iOS acceleration.
	CoreMLProviderBackend ProviderBackend = "coreml"
)

// CoreMLProvider implements the ExecutionProvider interface.
type CoreMLProvider struct {
	options CoreMLOptions
}

// CoreMLOptions contains arguments for the CoreML provider.
// See: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
type CoreMLOptions struct {
	// Create an MLProgram format model. Requires Core ML 5 or later (iOS 15+ or macOS 12+).
	// NeuralNetwork: Create a NeuralNetwork format model. Requires Core ML 3 or later (iOS 13+ or
	// macOS 10.15+).
	// Default: NeuralNetwork
	ModelFormat string `json:"modelFormat"                        yaml:"modelFormat"`
	// Limit CoreML to running on CPU only.
	// CPUAndNeuralEngine: Enable CoreML EP for Apple devices with a compatible Apple Neural Engine
	// (ANE).
	// CPUAndGPU: Enable CoreML EP for Apple devices with a compatible GPU.
	// ALL: Enable CoreML EP for all compatible Apple devices.
	// Default: ALL
	MLComputeUnits string `json:"mlComputeUnits"                     yaml:"mlComputeUnits"`
	// Only allow the CoreML EP to take nodes with inputs that have static shapes. By default the
	// CoreML EP will also allow inputs with dynamic shapes, however performance may be negatively
	// impacted by inputs
	// with dynamic shapes.
	// 0: Allow the CoreML EP to take nodes with inputs that have dynamic shapes.
	// 1: Only allow the CoreML EP to take nodes with inputs that have static shapes.
	// Default: 0
	RequireStaticInputShapes int `json:"requireStaticInputShapes"           yaml:"requireStaticInputShapes"`
	// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a Loop, Scan
	// or If operator).
	// 0: Disable CoreML EP to run on a subgraph in the body of a control flow operator.
	// 1: Enable CoreML EP to run on a subgraph in the body of a control flow operator.
	// Default: 0
	EnableOnSubgraphs int `json:"enableOnSubgraphs"                  yaml:"enableOnSubgraphs"`
	// This feature is available since macOS>=10.15 or iOS>=18.0. This process can affect the model
	// loading time and the prediction latency. Use this option to tailor the specialization strategy
	// for your model.
	// Navigate to Apple Doc for more information.
	// Default: Default
	SpecializationStrategy string `json:"specializationStrategy"             yaml:"specializationStrategy"`
	// Profile the Core ML MLComputePlan. This logs the hardware each operator is dispatched to and
	// the estimated execution time. Intended for developer usage but provides useful diagnostic
	// information if performance is
	// not as expected.
	// 0: Disable profile.
	// 1: Enable profile.
	// Default: 0
	ProfileComputePlan int `json:"profileComputePlan"                 yaml:"profileComputePlan"`
	// Please refer to Apple Doc for more information.
	// 0: Use float32 data type to accumulate data.
	// 1: Use low precision data(float16) to accumulate data.
	// Default: 0
	AllowLowPrecisionAccumulationOnGPU int `json:"allowLowPrecisionAccumulationOnGPU" yaml:"allowLowPrecisionAccumulationOnGPU"`
	// The path to the directory where the Core ML model cache is stored. CoreML EP will compile the
	// captured subgraph to CoreML format graph and saved to disk. For the given model, if caching is
	// not enabled, CoreML EP will compile and save to disk every time, which may cost significant
	// time (even minutes) for
	// a complicated model. By providing a cache path the CoreML format model can be reused.
	// Default: Cache disabled
	ModelCacheDirectory string `json:"modelCacheDirectory"                yaml:"modelCacheDirectory"`
}

func (CoreMLOptions) isProviderOptions() {}

// Backend returns the backend of the CUDA provider.
func (p *CoreMLProvider) Backend() ProviderBackend {
	return CUDAProviderBackend
}

// Options returns the options of the CUDA provider.
func (p *CoreMLProvider) Options() ProviderOptions {
	return p.options
}

// Execute executes the CUDA provider.
func (p *CoreMLProvider) Execute(input []byte) ([]byte, error) {
	// CUDA execution logic
	return input, nil
}

// NewCoreMLProvider creates a new CoreML provider.
func NewCoreMLProvider(options CoreMLOptions) *CoreMLProvider {
	return &CoreMLProvider{
		options: options,
	}
}

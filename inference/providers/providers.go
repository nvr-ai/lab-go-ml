package providers

// Provider represents different ONNX Runtime execution providers
type Provider string

const (
	// CPUExecutionProvider uses CPU for inference
	CPUExecutionProvider Provider = "cpu"

	// CUDAExecutionProvider uses NVIDIA CUDA for GPU acceleration
	CUDAExecutionProvider Provider = "cuda"

	// TensorRTExecutionProvider uses NVIDIA TensorRT for optimized inference
	TensorRTExecutionProvider Provider = "tensorrt"

	// DNNLExecutionProvider uses Intel DNNL (oneDNN) for CPU optimization
	DNNLExecutionProvider Provider = "dnnl"

	// CoreMLExecutionProvider uses Apple CoreML for macOS/iOS acceleration
	CoreMLExecutionProvider Provider = "coreml"

	// OpenVINOExecutionProvider uses Intel OpenVINO for inference optimization
	OpenVINOExecutionProvider Provider = "openvino"
)

// GetExecutionProviders returns the list of enabled execution providers in priority order
//
// This method extracts execution provider configuration from both legacy flags
// and modern OptimizationConfig for backward compatibility.
//
// Returns:
//   - []ExecutionProviderConfig: Enabled execution providers in priority order
func GetExecutionProviders(c Config) []ExecutionProviderConfig {
	var providers []ExecutionProviderConfig
	panic("not implemented")
	// // Handle legacy configuration flags
	// if c.OptimizationConfig.ExecutionProviders.Provider == CoreMLExecutionProvider {
	// 	providers = append(providers, ExecutionProviderConfig{
	// 		Provider: CoreMLExecutionProvider,
	// 		Options:  map[string]string{},
	// 		Priority: 10,
	// 		Enabled:  true,
	// 	})
	// }

	// if c.UseOpenVINO {
	// 	providers = append(providers, ExecutionProviderConfig{
	// 		Provider: OpenVINOExecutionProvider,
	// 		Options:  map[string]string{},
	// 		Priority: 8,
	// 		Enabled:  true,
	// 	})
	// }

	// // Add providers from OptimizationConfig
	// if c.OptimizationConfig != nil {
	// 	for _, provider := range c.OptimizationConfig.ExecutionProviders {
	// 		if provider.Enabled {
	// 			providers = append(providers, provider)
	// 		}
	// 	}
	// }

	// // Add default CPU provider if no others specified
	// if len(providers) == 0 {
	// 	providers = append(providers, ExecutionProviderConfig{
	// 		Provider: CPUExecutionProvider,
	// 		Options:  map[string]string{},
	// 		Priority: 1,
	// 		Enabled:  true,
	// 	})
	// }

	return providers
}

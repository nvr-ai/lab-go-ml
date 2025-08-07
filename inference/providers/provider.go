package providers

// ProviderBackend represents different ONNX Runtime execution providers
type ProviderBackend string

const (
	// CPUProviderBackend uses CPU for inference
	CPUProviderBackend ProviderBackend = "cpu"

	// CUDAProviderBackend uses NVIDIA CUDA for GPU acceleration
	CUDAProviderBackend ProviderBackend = "cuda"

	// TensorRTProviderBackend uses NVIDIA TensorRT for optimized inference
	TensorRTProviderBackend ProviderBackend = "tensorrt"

	// DNNLProviderBackend uses Intel DNNL (oneDNN) for CPU optimization
	DNNLProviderBackend ProviderBackend = "dnnl"

	// CoreMLProviderBackend uses Apple CoreML for macOS/iOS acceleration
	CoreMLProviderBackend ProviderBackend = "coreml"

	// OpenVINOProviderBackend uses Intel OpenVINO for inference optimization
	OpenVINOProviderBackend ProviderBackend = "openvino"
)

// Provider represents the contract that all execution providers must implement.
type Provider[T any] struct {
	Backend  ProviderBackend
	Options  map[string]string
	Priority int
	Enabled  bool
}

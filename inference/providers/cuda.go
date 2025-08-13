// Package providers - CPU based execution provider.
package providers

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	// CUDAProviderBackend uses NVIDIA CUDA for inference optimization.
	CUDAProviderBackend ProviderBackend = "cuda"
)

// CUDAProvider implements the ExecutionProvider interface.
type CUDAProvider struct {
	options CUDAOptions
}

// CUDAOptions contains arguments for the CUDA provider.
// See:
// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
type CUDAOptions struct {
	// The device ID.
	DeviceID int `json:"deviceID"                      yaml:"deviceID"`
	// Defines the compute stream for the inference to run on. It implicitly sets the
	// has_user_compute_stream option. This cannot be used in combination with an external allocator.
	UserComputeStream string `json:"userComputeStream"             yaml:"userComputeStream"`
	// Whether to do copies in the default stream or use separate streams. The recommended setting is
	// true. If false, there are race conditions and possibly better performance.
	DoCopyInDefaultStream bool `json:"doCopyInDefaultStream"         yaml:"doCopyInDefaultStream"`
	// Uses the same CUDA stream for all threads of the CUDA EP. This is implicitly enabled by
	// has_user_compute_stream, enable_cuda_graph or when using an external allocator.
	UseEPLevelUnifiedStream bool `json:"useEPLevelUnifiedStream"       yaml:"useEPLevelUnifiedStream"`
	// The size limit of the device memory arena in bytes. This size limit is only for the execution
	// provider's arena. The total device memory usage may be higher.
	GPUMemLimit int64 `json:"gpuMemLimit"                   yaml:"gpuMemLimit"`
	// The strategy for extending the device memory arena.
	// 0: kNextPowerOfTwo - subsequent extensions extend by larger amounts (multiplied by powers of
	// two)
	// 1: kSameAsRequested - extend by the requested amount
	ArenaExtendStrategy int `json:"arenaExtendStrategy"           yaml:"arenaExtendStrategy"`
	// The type of search done for cuDNN convolution algorithms.
	// 0: EXHAUSTIVE - expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
	// 1: HEURISTIC - lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
	// 2: DEFAULT - default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	CudnnConvAlgoSearch int `json:"cudnnConvAlgoSearch"           yaml:"cudnnConvAlgoSearch"`
	// Check tuning performance for convolution heavy models for details on what this flag does.
	CudnnConvUseMaxWorkspace int `json:"cudnnConvUseMaxWorkspace"      yaml:"cudnnConvUseMaxWorkspace"`
	// Check convolution input padding in the CUDA EP for details on what this flag does.
	CudnnConv1dPadToNC1d int `json:"cudnnConv1dPadToNC1d"          yaml:"cudnnConv1dPadToNC1d"`
	// Check using CUDA Graphs in the CUDA EP for details on what this flag does.
	EnableCudaGraph int `json:"enableCudaGraph"               yaml:"enableCudaGraph"`
	// Whether to use strict mode in SkipLayerNormalization cuda implementation. The default and
	// recommended setting is false.
	// If enabled, accuracy improvement and performance drop can be expected.
	EnableSkipLayerNormStrictMode int `json:"enableSkipLayerNormStrictMode" yaml:"enableSkipLayerNormStrictMode"`
	// TF32 is a math mode available on NVIDIA GPUs since Ampere. It allows certain float32 matrix
	// multiplications
	// and convolutions to run much faster on tensor cores with TensorFloat-32 reduced precision.
	UseTF32 int `json:"useTF32"                       yaml:"useTF32"`
	// External allocator address for GPU memory allocation.
	GPUExternalAlloc string `json:"gpuExternalAlloc"              yaml:"gpuExternalAlloc"`
	// External allocator address for GPU memory deallocation.
	GPUExternalFree string `json:"gpuExternalFree"               yaml:"gpuExternalFree"`
	// External allocator address for GPU cache emptying.
	GPUExternalEmptyCache string `json:"gpuExternalEmptyCache"         yaml:"gpuExternalEmptyCache"`
	// If this option is enabled, the execution provider prefers NHWC operators over NCHW.
	// Necessary layout transformations will be applied to the model automatically.
	PreferNHWC int `json:"preferNHWC"                    yaml:"preferNHWC"`
}

// ToNativeProviderOptions converts the CUDA options to a CUDA provider options.
// This is used to pass the options to the CUDA provider.
func (o *CUDAOptions) ToNativeProviderOptions() (*ort.CUDAProviderOptions, error) {
	opts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return nil, err
	}

	opts.Update(map[string]string{
		"deviceID":                      fmt.Sprintf("%d", o.DeviceID),
		"userComputeStream":             fmt.Sprintf("%s", o.UserComputeStream),
		"doCopyInDefaultStream":         fmt.Sprintf("%t", o.DoCopyInDefaultStream),
		"useEPLevelUnifiedStream":       fmt.Sprintf("%t", o.UseEPLevelUnifiedStream),
		"gpuMemLimit":                   fmt.Sprintf("%d", o.GPUMemLimit),
		"arenaExtendStrategy":           fmt.Sprintf("%d", o.ArenaExtendStrategy),
		"cudnnConvAlgoSearch":           fmt.Sprintf("%d", o.CudnnConvAlgoSearch),
		"cudnnConvUseMaxWorkspace":      fmt.Sprintf("%d", o.CudnnConvUseMaxWorkspace),
		"cudnnConv1dPadToNC1d":          fmt.Sprintf("%d", o.CudnnConv1dPadToNC1d),
		"enableCudaGraph":               fmt.Sprintf("%d", o.EnableCudaGraph),
		"enableSkipLayerNormStrictMode": fmt.Sprintf("%d", o.EnableSkipLayerNormStrictMode),
		"useTF32":                       fmt.Sprintf("%d", o.UseTF32),
		"gpuExternalAlloc":              fmt.Sprintf("%s", o.GPUExternalAlloc),
		"gpuExternalFree":               fmt.Sprintf("%s", o.GPUExternalFree),
		"gpuExternalEmptyCache":         fmt.Sprintf("%s", o.GPUExternalEmptyCache),
		"preferNHWC":                    fmt.Sprintf("%d", o.PreferNHWC),
	})

	return opts, nil
}

// isProviderOptions is a marker function to ensure the options are valid.
func (CUDAOptions) isProviderOptions() {}

// Backend returns the backend of the CUDA provider.
func (p *CUDAProvider) Backend() ProviderBackend {
	return CUDAProviderBackend
}

// Options returns the options of the CUDA provider.
func (p *CUDAProvider) Options() ProviderOptions {
	return p.options
}

// Execute executes the CUDA provider.
func (p *CUDAProvider) Execute(input []byte) ([]byte, error) {
	// CUDA execution logic
	return input, nil
}

// NewCUDAProvider creates a new CUDA provider.
func NewCUDAProvider(args CUDAOptions) *CUDAProvider {
	return &CUDAProvider{
		options: args,
	}
}

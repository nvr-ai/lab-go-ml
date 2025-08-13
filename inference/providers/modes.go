package providers

// ProviderMode represents the mode of the provider.
type ProviderMode string

const (
	// ProviderModeCPU uses CPU for inference.
	ProviderModeCPU ProviderMode = "cpu"

	// ProviderModeGPU uses GPU for inference.
	ProviderModeGPU ProviderMode = "gpu"
)

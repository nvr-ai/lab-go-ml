// Package providers - CPU based execution provider.
package providers

// CPUProvider represents the CPU execution provider
type CPUProvider struct {
	Backend  ProviderBackend
	Options  map[string]string
	Priority int
	Enabled  bool
}

// CUDAProvider represents the CUDA execution provider
type CUDAProvider struct {
	Backend  ProviderBackend
	Options  map[string]string
	Priority int
	Enabled  bool
}

// NewCPUProvider creates a new CPU provider
func NewCPUProvider() *Provider[CPUProvider] {
	return &Provider[CPUProvider]{
		Backend:  CPUProviderBackend,
		Options:  make(map[string]string),
		Priority: 1,
		Enabled:  true,
	}
}

// NewCUDAProvider creates a new CUDA provider
func NewCUDAProvider() *Provider[CUDAProvider] {
	return &Provider[CUDAProvider]{
		Backend:  CUDAProviderBackend,
		Options:  make(map[string]string),
		Priority: 1,
		Enabled:  true,
	}
}

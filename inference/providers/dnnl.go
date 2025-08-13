// Package providers - CPU based execution provider.
package providers

const (

	// DNNLProviderBackend uses Intel DNNL (oneDNN) for CPU optimization.
	DNNLProviderBackend ProviderBackend = "dnnl"
)

// DNNLProvider implements the ExecutionProvider interface.
type DNNLProvider struct {
	DeviceID int
}

// DNNLProviderOptions contains arguments for the DNNL provider.
type DNNLProviderOptions struct {
	DeviceID int
}

// isProviderOptions is a marker function to ensure the options are valid.
func (DNNLProviderOptions) isProviderOptions() {}

// Options returns the options of the DNNL provider.
func (p *DNNLProvider) Options() ProviderOptions {
	return DNNLProviderOptions{
		DeviceID: p.DeviceID,
	}
}

// Backend returns the backend of the DNNL provider.
func (p *DNNLProvider) Backend() ProviderBackend {
	return DNNLProviderBackend
}

// Execute executes the DNNL provider.
func (p *DNNLProvider) Execute(input []byte) ([]byte, error) {
	// DNNL execution logic
	return input, nil
}

// NewDNNLProvider creates a new DNNL provider.
func NewDNNLProvider(args DNNLProviderOptions) *DNNLProvider {
	return &DNNLProvider{
		DeviceID: args.DeviceID,
	}
}

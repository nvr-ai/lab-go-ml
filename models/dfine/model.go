// Package dfine - D-FINE model.
package dfine

import (
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/model"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// Options is the options for the D-FINE model.
type Options struct {
	Path    string                 `json:"path" yaml:"path"`
	NMS     *postprocess.NMSConfig `json:"nms" yaml:"nms"`
	Inputs  []string               `json:"inputs" yaml:"inputs"`
	Outputs []string               `json:"outputs" yaml:"outputs"`
	Shapes  []images.Rect          `json:"shapes" yaml:"shapes"`
}

// DFINE is the instance of the D-FINE model.
type DFINE struct {
	options Options
}

// Options returns the options for the RF-DETR model.
//
// Returns:
//   - The options for the D-FINE model.
func (m *DFINE) Options() model.Options {
	return m.options
}

// NewModel creates a new model.
//
// Arguments:
//   - args: The arguments for creating a new model.
//
// Returns:
//   - The model.
func NewModel(args model.NewModelArgs) (*DFINE, error) {
	return &DFINE{
		options: Options{
			Path:    args.Path,
			NMS:     args.NMS,
			Inputs:  args.Inputs,
			Outputs: args.Outputs,
			Shapes:  args.Shapes,
		},
	}, nil
}

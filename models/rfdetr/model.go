// Package rfdetr - RF-DETR model.
package rfdetr

import (
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/model"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// RFDETR is the instance of the RF-DETR model.
type RFDETR struct {
	options Options
}

// Options is the options for the RF-DETR model.
type Options struct {
	Name    model.Name             `json:"name" yaml:"name"`
	Family  model.Family           `json:"family" yaml:"family"`
	Path    string                 `json:"path" yaml:"path"`
	NMS     *postprocess.NMSConfig `json:"nms" yaml:"nms"`
	Inputs  []string               `json:"inputs" yaml:"inputs"`
	Outputs []string               `json:"outputs" yaml:"outputs"`
	Shapes  []images.Rect          `json:"shapes" yaml:"shapes"`
}

// Options returns the options for the RF-DETR model.
//
// Returns:
//   - The options for the RF-DETR model.
func (m *RFDETR) Options() model.Options {
	return m.options
}

// NewModel creates a new model.
//
// Arguments:
//   - args: The arguments for creating a new model.
//
// Returns:
//   - The model.
func NewModel(args model.NewModelArgs) (*RFDETR, error) {
	return &RFDETR{
		options: Options{
			Name:    model.ModelNameRFDETR,
			Family:  model.ModelFamilyVOC,
			Path:    args.Path,
			NMS:     args.NMS,
			Inputs:  args.Inputs,
			Outputs: args.Outputs,
			Shapes:  args.Shapes,
		},
	}, nil
}

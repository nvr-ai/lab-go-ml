// Package yolov4 - YOLOv4 model.
package yolov4

import (
	"fmt"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/model"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// NewModelArgs is the arguments for creating a new YOLOv4 model.
type NewModelArgs struct {
	Name    model.Name             `json:"name" yaml:"name"`
	Family  model.Family           `json:"family" yaml:"family"`
	Path    string                 `json:"path" yaml:"path"`
	NMS     *postprocess.NMSConfig `json:"nms" yaml:"nms"`
	Inputs  []string               `json:"inputs" yaml:"inputs"`
	Outputs []string               `json:"outputs" yaml:"outputs"`
	Shapes  []images.Rect          `json:"shapes" yaml:"shapes"`
}

// Options is the options for the YOLOv4 model.
type Options struct {
	Name    model.Name             `json:"name" yaml:"name"`
	Family  model.Family           `json:"family" yaml:"family"`
	Path    string                 `json:"path" yaml:"path"`
	NMS     *postprocess.NMSConfig `json:"nms" yaml:"nms"`
	Inputs  []string               `json:"inputs" yaml:"inputs"`
	Outputs []string               `json:"outputs" yaml:"outputs"`
	Shapes  []images.Rect          `json:"shapes" yaml:"shapes"`
}

// YOLOv4 is the instance of the YOLOv4 model.
type YOLOv4 struct {
	options Options
}

// Options returns the options for the RF-DETR model.
//
// Returns:
//   - The options for the YOLOv4 model.
func (m *YOLOv4) Options() model.Options {
	return m.options
}

// NewModel creates a new model.
//
// Arguments:
//   - args: The arguments for creating a new model.
//
// Returns:
//   - The model.
func NewModel(args model.NewModelArgs) (*YOLOv4, error) {
	if len(args.Inputs) == 0 {
		return nil, fmt.Errorf("NewModel requires inputs to be set")
	}

	if len(args.Outputs) == 0 {
		return nil, fmt.Errorf("NewModel requires outputs to be set")
	}

	if len(args.Shapes) == 0 {
		return nil, fmt.Errorf("NewModel requires shapes to be set")
	}

	return &YOLOv4{
		options: Options{
			Name:    model.ModelNameYOLOv4,
			Family:  model.ModelFamilyYOLO,
			Path:    args.Path,
			NMS:     args.NMS,
			Inputs:  args.Inputs,
			Outputs: args.Outputs,
			Shapes:  args.Shapes,
		},
	}, nil
}

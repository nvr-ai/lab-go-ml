// Package models - registry for models.
package models

import (
	"fmt"

	"github.com/nvr-ai/go-ml/models/dfine"
	"github.com/nvr-ai/go-ml/models/model"
	"github.com/nvr-ai/go-ml/models/postprocess"
	"github.com/nvr-ai/go-ml/models/rfdetr"
	"github.com/nvr-ai/go-ml/models/yolov4"
)

// NewModel creates a new detection model instance based on the specified model type.
//
// This factory function serves as the primary entry point for model creation,
// routing requests to the appropriate model-specific constructors while
// providing a unified interface for model instantiation across the system.
//
// The factory pattern ensures that model creation is centralized, making it
// easier to add new model types and maintain consistent initialization logic.
//
// Arguments:
//   - args: Configuration parameters specifying the model type and location.
//
// Returns:
//   - model.Model: A fully configured model instance implementing the Model interface.
//
// - error: An error if model creation fails, the model type is unsupported, or validation errors
// occur.
//
// Example:
//
// ```go
//
//	args := NewModelArgs{
//	    Name: model.ModelNameRFDETR,
//	    Path: "/models/rfdetr_coco.onnx",
//	}
//
// detectionModel, err := NewModel(args)
//
//	if err != nil {
//	    log.Fatalf("Failed to create detection model: %v", err)
//	}
//
// fmt.Printf("Created %s model from family %s\n", detectionModel.GetName(),
// detectionModel.GetFamily())
//
//	dfineArgs := NewModelArgs{
//	    Name: model.ModelNameDFINE,
//	    Path: "/models/dfine_coco.onnx",
//	}
//
// dfineModel, err := NewModel(dfineArgs)
//
//	if err != nil {
//	    log.Fatalf("Failed to create DFINE model: %v", err)
//	}
//
// ```
func NewModel(args model.NewModelArgs) (model.Model, error) {
	switch args.Name {
	case model.ModelNameRFDETR:
		m, err := rfdetr.NewModel(model.NewModelArgs{Path: args.Path})
		if err != nil {
			return nil, err
		}
		return m, nil
	case model.ModelNameDFINE:
		m, err := dfine.NewModel(model.NewModelArgs{Path: args.Path})
		if err != nil {
			return nil, err
		}
		return m, nil
	case model.ModelNameYOLOv4:
		m, err := yolov4.NewModel(model.NewModelArgs{Path: args.Path})
		if err != nil {
			return nil, err
		}
		return m, nil
	default:
		return nil, fmt.Errorf("unsupported model name: %s", args.Name)
	}
}

type _model struct {
	options model.NewModelArgs
	model   model.Model
}

func (m *_model) Options() model.NewModelArgs {
	return m.options
}

func (m *_model) PostProcess(output []float32, config *postprocess.NMSConfig) []postprocess.Result {
	return m.model.PostProcess(output, config)
}

// Package model - Definitions for model output class styles and sets.
package model

import (
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// Family is the family of models.
type Family string

const (
	// ModelFamilyCOCO is the COCO model family.
	ModelFamilyCOCO Family = "coco"
	// ModelFamilyYOLO is the YOLO model family.
	ModelFamilyYOLO Family = "yolo"
	// ModelFamilyTF is the TensorFlow model family.
	ModelFamilyTF Family = "tf"
	// ModelFamilyVOC is the Pascal VOC model family.
	ModelFamilyVOC Family = "voc"
)

// Name is the unique identifier of a model.
type Name string

const (
	// ModelNameRFDETR is the name of the RF-DETR model.
	ModelNameRFDETR Name = "rfdetr"
	// ModelNameDFINE is the name of the D-FINE model.
	ModelNameDFINE Name = "dfine"
	// ModelNameYOLOv4 is the name of the YOLOv4 model.
	ModelNameYOLOv4 Name = "yolov4"
)

// Config is a model with a family and path for loading.
type Config struct {
	Family              Family
	Path                string
	ConfidenceThreshold float32
	NMS                 *postprocess.NMSConfig
	Inputs              []string
	Outputs             []string
	Shapes              []images.Rect
}

// BaseModel is the base model for all models.
type BaseModel struct {
	Name    Name
	Family  Family
	Path    string
	Inputs  []images.Rect
	Outputs []images.Rect
}

// Options is a marker interface for model-specific options.
type Options interface {
	IsOptions()
}

// Model is a model with a family and path for loading.
type Model interface {
	Options() BaseModel
	PreProcess(input []images.Image) []float32
	PostProcess(output []float32, config *postprocess.NMSConfig) []postprocess.Result
}

// NewModelArgs is the arguments for creating a new model.
type NewModelArgs struct {
	Name    Name                   `json:"name" yaml:"name"`
	Path    string                 `json:"path" yaml:"path"`
	NMS     *postprocess.NMSConfig `json:"nms" yaml:"nms"`
	Family  Family                 `json:"family" yaml:"family"`
	Inputs  []string               `json:"inputs" yaml:"inputs"`
	Outputs []string               `json:"outputs" yaml:"outputs"`
}

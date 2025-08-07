// Package models - Definitions for model output class styles and sets.
package models

// OutputClassGeneration identifies the naming convention / dataset.
type OutputClassGeneration string

const (
	// ModelTypeCOCO is the 80 COCO classes + background.
	ModelTypeCOCO OutputClassGeneration = "coco"
	// ModelTypeYOLO is the 80 COCO classes, no background (no background class).
	ModelTypeYOLO OutputClassGeneration = "yolo"
	// ModelTypeTF is the same 80 classes + background.
	ModelTypeTF OutputClassGeneration = "tf"
	// ModelTypeVOC is the 20 classes + background.
	ModelTypeVOC OutputClassGeneration = "voc"
)

// ModelFamily is the family of models.
type ModelFamily string

const (
	// ModelFamilyCOCO is the COCO model family.
	ModelFamilyCOCO ModelFamily = "coco"
	// ModelFamilyYOLO is the YOLO model family.
	ModelFamilyYOLO ModelFamily = "yolo"
	// ModelFamilyTF is the TensorFlow model family.
	ModelFamilyTF ModelFamily = "tf"
	// ModelFamilyVOC is the Pascal VOC model family.
	ModelFamilyVOC ModelFamily = "voc"
)

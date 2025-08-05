package onnx

import (
	"fmt"
	"image"
	"testing"

	"github.com/nvr-ai/go-ml/util"
	"github.com/stretchr/testify/assert"
)

func TestNewONNXDetector(t *testing.T) {
	var err error

	objectDetector, err := NewONNXDetector(Config{
		ModelPath:           "../yolov3u.onnx",
		InputShape:          image.Point{X: 416, Y: 416},
		ConfidenceThreshold: 0.5,
		NMSThreshold:        0.5,
		RelevantClasses:     []string{"person", "car", "truck", "bus", "motorcycle", "bicycle"},
	})

	assert.NoErrorf(t, err, "Failed to create ONNX detector: %v", err)

	fmt.Printf("ONNX Detector initialized with model: %s\n", objectDetector.modelPath)

	images, err := util.LoadDirectoryImageFiles("/corpus/images/videos/freeway-view-22-seconds-1080p.mp4")
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}

	assert.NoErrorf(t, err, "Failed to load images: %v", err)

	fmt.Printf("Loaded %d images for testing\n", len(images))

}

package onnx

import (
	"fmt"
	"image"
	"strings"
	"testing"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/util"
	"github.com/stretchr/testify/assert"
)

func TestNewONNXDetectorTest(t *testing.T) {
	var err error

	detector, err := NewSession(Config{
		ModelPath:           "../data/yolov8n.onnx",
		InputShape:          image.Point{X: 416, Y: 416},
		ConfidenceThreshold: 0.5,
		NMSThreshold:        0.5,
		RelevantClasses:     []string{"person", "car", "truck", "bus", "motorcycle", "bicycle"},
	})
	if err != nil {
		// Check if it's a library not found error
		if strings.Contains(err.Error(), "ONNX Runtime library not found") {
			t.Skipf("Skipping ONNX Runtime test - library not available: %v", err)
			return
		}
		t.Fatalf("Failed to create ONNX detector: %v", err)
	}
	defer detector.Close() // Properly clean up resources

	fmt.Printf("ONNX Detector initialized with model: %+v\n", detector.Session)

	imgs, err := util.LoadDirectoryImageFiles("../../../ml/corpus/images/videos/freeway-view-22-seconds-1080p.mp4")
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}

	assert.NoErrorf(t, err, "Failed to load images: %v", err)

	fmt.Printf("Loaded %d images for testing\n", len(imgs))

	for _, frame := range imgs {
		resized, err := images.ResizeWebPToImage(frame.Data, 240, 240)
		if err != nil {
			t.Fatalf("Failed to resize JPEG: %v", err)
		}

		// Prepare the input for the ONNX model.
		err = PrepareInput(resized, detector.Input)
		if err != nil {
			t.Fatalf("Failed to prepare input: %v", err)
		}

		// Runs the session, updating the contents of the output tensors on success.
		err = detector.Session.Run()
		if err != nil {
			t.Fatalf("Failed to run session: %v", err)
		}

		detections := ProcessInferenceOutput(detector.Output.GetData(), resized.Bounds().Canon().Dx(), resized.Bounds().Canon().Dy())
		fmt.Printf("Detected %d objects\n", len(detections))
	}
}

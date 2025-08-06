package onnx

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/nvr-ai/go-ml/common"
	"github.com/nvr-ai/go-ml/models/dfine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ort "github.com/yalue/onnxruntime_go"
)

// TestBoundingBoxString verifies that bounding box string formatting works correctly.
//
// This test ensures that the String() method properly formats bounding box
// information with the expected precision and format, making debugging easier.
//
// @example
// go test -v -run TestBoundingBoxString
func TestBoundingBoxString(t *testing.T) {
	// Welcome! This test helps ensure our detection results are displayed clearly.
	tests := []struct {
		name     string
		box      common.BoundingBox
		expected string
	}{
		{
			name: "person detection with high confidence",
			box: common.BoundingBox{
				Label:      "person",
				Confidence: 0.95,
				X1:         100.123,
				Y1:         200.456,
				X2:         300.789,
				Y2:         400.012,
			},
			expected: "Object person (confidence 0.950000): (100.12, 200.46), (300.79, 400.01)",
		},
		{
			name: "car detection with medium confidence",
			box: common.BoundingBox{
				Label:      "car",
				Confidence: 0.75,
				X1:         0,
				Y1:         0,
				X2:         50.5,
				Y2:         75.5,
			},
			expected: "Object car (confidence 0.750000): (0.00, 0.00), (50.50, 75.50)",
		},
		{
			name: "edge case with very small confidence",
			box: common.BoundingBox{
				Label:      "bicycle",
				Confidence: 0.001,
				X1:         -10,
				Y1:         -10,
				X2:         10,
				Y2:         10,
			},
			expected: "Object bicycle (confidence 0.001000): (-10.00, -10.00), (10.00, 10.00)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.box.String()
			assert.Equal(t, tt.expected, result,
				"String representation should match expected format")
		})
	}
}

// TestBoundingBoxToRect verifies the conversion from floating-point to integer coordinates.
//
// This test ensures that our bounding boxes can be properly converted to
// image.Rectangle for use with Go's standard image processing functions.
//
// @example
// go test -v -run TestBoundingBoxToRect
func TestBoundingBoxToRect(t *testing.T) {
	// Let's make sure our coordinate conversions work perfectly!
	tests := []struct {
		name     string
		box      common.BoundingBox
		expected image.Rectangle
	}{
		{
			name: "standard conversion",
			box: common.BoundingBox{
				X1: 10.4,
				Y1: 20.6,
				X2: 100.8,
				Y2: 200.2,
			},
			expected: image.Rect(10, 20, 100, 200),
		},
		{
			name: "handles negative coordinates",
			box: common.BoundingBox{
				X1: -10.5,
				Y1: -20.5,
				X2: 50.5,
				Y2: 60.5,
			},
			expected: image.Rect(-10, -20, 50, 60),
		},
		{
			name: "ensures canonical form",
			box: common.BoundingBox{
				X1: 100,
				Y1: 100,
				X2: 0,
				Y2: 0,
			},
			expected: image.Rect(0, 0, 100, 100),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.box.ToRect().Canon()
			assert.Equal(t, tt.expected, result,
				"Rectangle conversion should handle rounding and canonicalization")
		})
	}
}

// TestBoundingBoxIntersection verifies intersection area calculations.
//
// This test ensures that we can accurately calculate the overlapping area
// between two bounding boxes, which is crucial for NMS operations.
//
// @example
// go test -v -run TestBoundingBoxIntersection
func TestBoundingBoxIntersection(t *testing.T) {
	// Testing intersection calculations - essential for removing duplicate detections!
	tests := []struct {
		name         string
		box1         common.BoundingBox
		box2         common.BoundingBox
		expectedArea float32
	}{
		{
			name:         "50% overlap",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:         common.BoundingBox{X1: 50, Y1: 50, X2: 150, Y2: 150},
			expectedArea: 2500, // 50x50 overlap
		},
		{
			name:         "complete overlap",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:         common.BoundingBox{X1: 25, Y1: 25, X2: 75, Y2: 75},
			expectedArea: 2500, // 50x50 inner box
		},
		{
			name:         "no overlap",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 50, Y2: 50},
			box2:         common.BoundingBox{X1: 100, Y1: 100, X2: 150, Y2: 150},
			expectedArea: 0,
		},
		{
			name:         "edge touching",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 50, Y2: 50},
			box2:         common.BoundingBox{X1: 50, Y1: 0, X2: 100, Y2: 50},
			expectedArea: 0, // Touching edges don't count as intersection
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.box1.Intersection(&tt.box2)
			assert.Equal(t, tt.expectedArea, result,
				"Intersection area should be calculated correctly")

			// Verify commutativity
			reverseResult := tt.box2.Intersection(&tt.box1)
			assert.Equal(t, result, reverseResult,
				"Intersection should be commutative")
		})
	}
}

// TestBoundingBoxUnion verifies union area calculations.
//
// This test ensures accurate calculation of the combined area of two bounding
// boxes, accounting for their intersection to avoid double-counting.
//
// @example
// go test -v -run TestBoundingBoxUnion
func TestBoundingBoxUnion(t *testing.T) {
	// Union calculations help us understand how much two detections overlap!
	tests := []struct {
		name         string
		box1         common.BoundingBox
		box2         common.BoundingBox
		expectedArea float32
	}{
		{
			name:         "partial overlap",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:         common.BoundingBox{X1: 50, Y1: 50, X2: 150, Y2: 150},
			expectedArea: 17500, // 10000 + 10000 - 2500
		},
		{
			name:         "no overlap",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 50, Y2: 50},
			box2:         common.BoundingBox{X1: 100, Y1: 100, X2: 150, Y2: 150},
			expectedArea: 5000, // 2500 + 2500
		},
		{
			name:         "complete containment",
			box1:         common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:         common.BoundingBox{X1: 25, Y1: 25, X2: 75, Y2: 75},
			expectedArea: 10000, // Larger box area only
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.box1.Union(&tt.box2)
			assert.Equal(t, tt.expectedArea, result,
				"Union area should be calculated correctly")

			// Verify commutativity
			reverseResult := tt.box2.Union(&tt.box1)
			assert.Equal(t, result, reverseResult,
				"Union should be commutative")
		})
	}
}

// TestBoundingBoxIoU verifies Intersection over Union calculations.
//
// This test ensures accurate IoU calculation, which is critical for
// Non-Maximum Suppression to work correctly in object detection.
//
// @example
// go test -v -run TestBoundingBoxIoU
func TestBoundingBoxIoU(t *testing.T) {
	// IoU is the key metric for determining if two detections are the same object!
	tests := []struct {
		name        string
		box1        common.BoundingBox
		box2        common.BoundingBox
		expectedIoU float32
		tolerance   float32
	}{
		{
			name:        "identical boxes",
			box1:        common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:        common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			expectedIoU: 1.0,
			tolerance:   0.001,
		},
		{
			name:        "50% overlap",
			box1:        common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:        common.BoundingBox{X1: 50, Y1: 50, X2: 150, Y2: 150},
			expectedIoU: 0.1428, // 2500/17500
			tolerance:   0.001,
		},
		{
			name:        "no overlap",
			box1:        common.BoundingBox{X1: 0, Y1: 0, X2: 50, Y2: 50},
			box2:        common.BoundingBox{X1: 100, Y1: 100, X2: 150, Y2: 150},
			expectedIoU: 0.0,
			tolerance:   0.001,
		},
		{
			name:        "small box inside large box",
			box1:        common.BoundingBox{X1: 0, Y1: 0, X2: 100, Y2: 100},
			box2:        common.BoundingBox{X1: 40, Y1: 40, X2: 60, Y2: 60},
			expectedIoU: 0.04, // 400/10000
			tolerance:   0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.box1.IOU(&tt.box2)
			assert.InDelta(t, tt.expectedIoU, result, float64(tt.tolerance),
				"IoU should be within tolerance")

			// Verify commutativity
			reverseResult := tt.box2.IOU(&tt.box1)
			assert.InDelta(t, result, reverseResult, 0.0001,
				"IoU should be commutative")
		})
	}
}

// TestProcessDFINEOutput verifies the complete output processing pipeline.
//
// This test ensures that raw model outputs are correctly transformed into
// meaningful detections with proper confidence thresholding and NMS.
//
// @example
// go test -v -run TestProcessDFINEOutput
func TestProcessDFINEOutput(t *testing.T) {
	// Testing the heart of our detection system - let's make it rock solid!
	numQueries := 5
	numClasses := 80

	// Create mock outputs
	logits := make([]float32, numQueries*numClasses)
	boxes := make([]float32, numQueries*4)

	// Set up test detections
	// Detection 1: High confidence person at (100, 100) with size 50x50
	logits[0*numClasses+0] = 2.0 // High logit for person class
	boxes[0*4+0] = 0.2           // cx = 0.2 * width
	boxes[0*4+1] = 0.2           // cy = 0.2 * height
	boxes[0*4+2] = 0.1           // w = 0.1 * width
	boxes[0*4+3] = 0.1           // h = 0.1 * height

	// Detection 2: Medium confidence car overlapping with detection 1
	logits[1*numClasses+2] = 1.0 // Medium logit for car class
	boxes[1*4+0] = 0.22          // Slight offset from detection 1
	boxes[1*4+1] = 0.22
	boxes[1*4+2] = 0.12
	boxes[1*4+3] = 0.12

	// Detection 3: Low confidence bicycle (should be filtered out)
	logits[2*numClasses+1] = -1.0 // Low logit
	boxes[2*4+0] = 0.5
	boxes[2*4+1] = 0.5
	boxes[2*4+2] = 0.1
	boxes[2*4+3] = 0.1

	// Detection 4: High confidence dog at different location
	logits[3*numClasses+16] = 2.5 // Very high logit for dog class
	boxes[3*4+0] = 0.7
	boxes[3*4+1] = 0.7
	boxes[3*4+2] = 0.15
	boxes[3*4+3] = 0.15

	originalWidth := 640
	originalHeight := 480
	confThreshold := float32(0.5)
	nmsThreshold := float32(0.5)

	detections := dfine.ProcessDFINEOutput(
		logits,
		boxes,
		numClasses,
		originalWidth,
		originalHeight,
		confThreshold,
		nmsThreshold,
	)

	// Verify results
	assert.GreaterOrEqual(t, len(detections), 2,
		"Should have at least 2 detections after filtering")
	assert.LessOrEqual(t, len(detections), 3,
		"Should have at most 3 detections after NMS")

	// Check that detections are sorted by confidence
	for i := 1; i < len(detections); i++ {
		assert.GreaterOrEqual(t, detections[i-1].Intersection(&detections[i]), detections[i].Confidence,
			"Detections should be sorted by confidence (descending)")
	}

	// Verify all detections meet confidence threshold
	for _, det := range detections {
		assert.GreaterOrEqual(t, det.Confidence, confThreshold,
			"All detections should meet confidence threshold")
	}

	// Verify bounding box coordinates are within image bounds
	for _, det := range detections {
		assert.GreaterOrEqual(t, det.X1, float32(0),
			"x1 should be within image bounds")
		assert.GreaterOrEqual(t, det.Y1, float32(0),
			"y1 should be within image bounds")
		assert.LessOrEqual(t, det.X2, float32(originalWidth),
			"x2 should be within image bounds")
		assert.LessOrEqual(t, det.Y2, float32(originalHeight),
			"y2 should be within image bounds")
	}
}

// TestExtractMultiScaleFeatures verifies multi-scale feature extraction.
//
// This test ensures that features are correctly extracted at multiple scales,
// simulating the output of a Feature Pyramid Network backbone.
//
// @example
// go test -v -run TestExtractMultiScaleFeatures
func TestExtractMultiScaleFeatures(t *testing.T) {
	// Feature extraction is crucial for D-FINE - let's test it thoroughly!
	// Create a test image
	width, height := 640, 480
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// Fill with a gradient pattern for testing
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r := uint8(float64(x) / float64(width) * 255)
			g := uint8(float64(y) / float64(height) * 255)
			b := uint8(128)
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}

	// Create mock session
	featStrides := []int{8, 16, 32}
	featChannels := []int{512, 1024, 2048}

	session := &dfine.DFINEModelSession{
		FeatStrides:  featStrides,
		FeatChannels: featChannels,
		FeatureMaps:  make([]*ort.Tensor[float32], len(featStrides)),
	}

	// Initialize feature tensors
	for i, stride := range featStrides {
		featHeight := height / stride
		featWidth := width / stride
		channels := featChannels[i]

		shape := ort.NewShape(1, int64(channels), int64(featHeight), int64(featWidth))
		tensor, err := ort.NewEmptyTensor[float32](shape)
		require.NoError(t, err, "Should create tensor successfully")
		session.FeatureMaps[i] = tensor
		defer tensor.Destroy()
	}

	// Extract features
	err := common.ExtractMultiScaleFeatures(img, session.FeatStrides, session.FeatChannels[0], session.FeatureMaps[0].GetData())
	assert.NoError(t, err, "Feature extraction should succeed")

	// Verify feature maps
	for i, tensor := range session.FeatureMaps {
		data := tensor.GetData()
		stride := featStrides[i]
		channels := featChannels[i]
		featHeight := height / stride
		featWidth := width / stride
		expectedSize := channels * featHeight * featWidth

		assert.Equal(t, expectedSize, len(data),
			"Feature tensor should have correct size")

		// Check that RGB channels contain normalized values
		rgbChannelSize := featHeight * featWidth
		for c := 0; c < 3 && c < channels; c++ {
			channelStart := c * rgbChannelSize
			channelEnd := channelStart + rgbChannelSize

			for idx := channelStart; idx < channelEnd; idx++ {
				assert.GreaterOrEqual(t, data[idx], float32(0),
					"Normalized values should be >= 0")
				assert.LessOrEqual(t, data[idx], float32(1),
					"Normalized values should be <= 1")
			}
		}
	}
}

// TestInitDFINESession verifies session initialization with various configurations.
//
// This test ensures that D-FINE sessions can be properly initialized with
// different image sizes and feature configurations.
//
// @example
// go test -v -run TestInitDFINESession
func TestInitDFINESession(t *testing.T) {
	// Session initialization is where it all begins - let's make it bulletproof!
	// Skip if ONNX Runtime is not available
	if _, err := os.Stat(getSharedLibPath()); os.IsNotExist(err) {
		t.Skip("ONNX Runtime library not found, skipping session tests")
	}

	// Create a minimal mock ONNX model for testing
	// In real tests, you would use a proper test model
	modelPath := filepath.Join(t.TempDir(), "test_model.onnx")

	tests := []struct {
		name         string
		width        int
		height       int
		featStrides  []int
		featChannels []int
		expectError  bool
	}{
		{
			name:         "standard 640x640 configuration",
			width:        640,
			height:       640,
			featStrides:  []int{8, 16, 32},
			featChannels: []int{512, 1024, 2048},
			expectError:  true, // Will error without valid model file
		},
		{
			name:         "HD 1920x1080 configuration",
			width:        1920,
			height:       1080,
			featStrides:  []int{8, 16, 32, 64},
			featChannels: []int{256, 512, 1024, 2048},
			expectError:  true,
		},
		{
			name:         "small 320x240 configuration",
			width:        320,
			height:       240,
			featStrides:  []int{8, 16},
			featChannels: []int{512, 1024},
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			session, err := dfine.InitDFINESession(
				modelPath,
				tt.width,
				tt.height,
				tt.featStrides,
				tt.featChannels,
			)

			if tt.expectError {
				assert.Error(t, err,
					"Should error without valid model file")
			} else {
				assert.NoError(t, err,
					"Session initialization should succeed")
				assert.NotNil(t, session,
					"Session should not be nil")

				// Verify session properties
				assert.Equal(t, len(tt.featStrides), len(session.FeatureMaps),
					"Should have correct number of feature maps")
				assert.Equal(t, tt.featStrides, session.FeatStrides,
					"Feature strides should match")
				assert.Equal(t, tt.featChannels, session.FeatChannels,
					"Feature channels should match")

				// Clean up
				session.Destroy()
			}
		})
	}
}

// TestMinMaxFunctions verifies the min and max utility functions.
//
// These simple functions are essential for clamping bounding box coordinates,
// so let's make sure they work perfectly!
//
// @example
// go test -v -run TestMinMaxFunctions
func TestMinMaxFunctions(t *testing.T) {
	// Even simple functions deserve thorough testing!
	tests := []struct {
		name   string
		a      float32
		b      float32
		expMin float32
		expMax float32
	}{
		{
			name:   "positive numbers",
			a:      3.14,
			b:      2.71,
			expMin: 2.71,
			expMax: 3.14,
		},
		{
			name:   "negative numbers",
			a:      -5.5,
			b:      -2.2,
			expMin: -5.5,
			expMax: -2.2,
		},
		{
			name:   "mixed signs",
			a:      -1.5,
			b:      1.5,
			expMin: -1.5,
			expMax: 1.5,
		},
		{
			name:   "equal values",
			a:      42.0,
			b:      42.0,
			expMin: 42.0,
			expMax: 42.0,
		},
		{
			name:   "infinity values",
			a:      float32(math.Inf(1)),
			b:      100,
			expMin: 100,
			expMax: float32(math.Inf(1)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			minResult := min(int(tt.a), int(tt.b))
			assert.Equal(t, tt.expMin, minResult,
				"min should return smaller value")

			maxResult := max(int(tt.a), int(tt.b))
			assert.Equal(t, tt.expMax, maxResult,
				"max should return larger value")

			// Verify commutativity
			assert.Equal(t, min(int(tt.b), int(tt.a)), minResult,
				"min should be commutative")
			assert.Equal(t, max(int(tt.b), int(tt.a)), maxResult,
				"max should be commutative")
		})
	}
}

// TestCOCOClasses verifies the COCO class list integrity.
//
// This test ensures that our COCO class list is complete and correctly ordered,
// which is crucial for proper object labeling.
//
// @example
// go test -v -run TestCOCOClasses
func TestCOCOClasses(t *testing.T) {
	// Let's verify our class list is ready for action!
	assert.Equal(t, 80, len(GetCOCOClasses()),
		"COCO dataset should have exactly 80 classes")

	// Verify some key classes are at correct indices
	expectedClasses := map[int]string{
		0:  "person",
		1:  "bicycle",
		2:  "car",
		16: "dog",
		17: "horse",
		79: "toothbrush",
	}

	for idx, expected := range expectedClasses {
		assert.Equal(t, expected, GetCOCOClasses()[idx],
			fmt.Sprintf("Class at index %d should be '%s'", idx, expected))
	}

	// Ensure no empty strings
	for i, class := range GetCOCOClasses() {
		assert.NotEmpty(t, class,
			fmt.Sprintf("Class at index %d should not be empty", i))
	}
}

// createTestImage creates a test image with specific patterns for testing.
//
// This helper function generates test images with known patterns that can
// be used to verify feature extraction and processing.
//
// Arguments:
// - width: Image width in pixels.
// - height: Image height in pixels.
// - pattern: Pattern type ("gradient", "checkerboard", "solid").
//
// Returns:
// - A generated test image.
//
// @example
// img := createTestImage(640, 480, "gradient")
// // Use img for testing feature extraction
func createTestImage(width, height int, pattern string) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	switch pattern {
	case "gradient":
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r := uint8(float64(x) / float64(width) * 255)
				g := uint8(float64(y) / float64(height) * 255)
				b := uint8(128)
				img.Set(x, y, color.RGBA{r, g, b, 255})
			}
		}
	case "checkerboard":
		squareSize := 32
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				if ((x/squareSize)+(y/squareSize))%2 == 0 {
					img.Set(x, y, color.RGBA{255, 255, 255, 255})
				} else {
					img.Set(x, y, color.RGBA{0, 0, 0, 255})
				}
			}
		}
	case "solid":
		draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{128, 128, 128, 255}},
			image.Point{}, draw.Src)
	default:
		// Default to solid gray
		draw.Draw(img, img.Bounds(), &image.Uniform{color.RGBA{128, 128, 128, 255}},
			image.Point{}, draw.Src)
	}

	return img
}

// BenchmarkProcessDFINEOutput benchmarks the output processing performance.
//
// This benchmark helps identify performance bottlenecks in the detection
// post-processing pipeline.
//
// @example
// go test -bench=BenchmarkProcessDFINEOutput -benchmem
func BenchmarkProcessDFINEOutput(b *testing.B) {
	// Performance matters! Let's measure how fast we can process detections.
	numQueries := 300 // Typical D-FINE output
	numClasses := 80

	// Create realistic mock outputs
	logits := make([]float32, numQueries*numClasses)
	boxes := make([]float32, numQueries*4)

	// Fill with random-ish data
	for i := range logits {
		logits[i] = float32(i%20 - 10) // Range -10 to 10
	}
	for i := range boxes {
		boxes[i] = float32(i%100) / 100.0 // Range 0 to 1
	}

	originalWidth := 1920
	originalHeight := 1080
	confThreshold := float32(0.5)
	nmsThreshold := float32(0.7)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dfine.ProcessDFINEOutput(
			logits,
			boxes,
			numClasses,
			originalWidth,
			originalHeight,
			confThreshold,
			nmsThreshold,
		)
	}
}

// BenchmarkBoundingBoxIoU benchmarks IoU calculation performance.
//
// This benchmark helps ensure IoU calculations remain fast even when
// processing many detections.
//
// @example
// go test -bench=BenchmarkBoundingBoxIoU -benchmem
func BenchmarkBoundingBoxIoU(b *testing.B) {
	// IoU is called many times during NMS - it needs to be fast!
	box1 := common.BoundingBox{X1: 10, Y1: 20, X2: 100, Y2: 200}
	box2 := common.BoundingBox{X1: 50, Y1: 60, X2: 150, Y2: 250}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = box1.IOU(&box2)
	}
}

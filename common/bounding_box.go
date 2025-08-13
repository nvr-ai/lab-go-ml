package common

import (
	"fmt"
	"image"

	"github.com/nfnt/resize"
)

// BoundingBox represents a bounding box with its label, confidence, and coordinates.
type BoundingBox struct {
	Label          string
	Confidence     float32
	X1, Y1, X2, Y2 float32
}

// This won't be entirely precise due to conversion to the integral rectangles
// from the image.Image library, but we're only using it to estimate which
// boxes are overlapping too much, so some imprecision should be OK.
func (b *BoundingBox) IOU(other *BoundingBox) float32 {
	return b.Intersection(other) / b.Union(other)
}

func (b *BoundingBox) String() string {
	return fmt.Sprintf("Object %s (confidence %f): (%f, %f), (%f, %f)",
		b.Label, b.Confidence, b.X1, b.Y1, b.X2, b.Y2)
}

// String formats the bounding box information for display.
//
// Returns:
// - A formatted string containing object class, confidence, and coordinates.
//
// @example
// box := BoundingBox{label: "person", confidence: 0.95, x1: 100, y1: 100, x2: 200, y2: 300}
// fmt.Println(box.String()) // Output: Object person (confidence 0.950000): (100.00, 100.00),
// (200.00, 300.00)

// ToRect converts the bounding box to an image.Rectangle.
//
// This method converts floating-point coordinates to integer coordinates
// suitable for image processing operations.
//
// Returns:
// - An image.Rectangle with canonicalized coordinates.
//
// @example
// box := BoundingBox{x1: 100.5, y1: 100.5, x2: 200.5, y2: 300.5}
// rect := box.ToRect()
// fmt.Printf("Rectangle: %v\n", rect) // Rectangle: (100,100)-(201,301)
func (b *BoundingBox) ToRect() image.Rectangle {
	return image.Rect(int(b.X1), int(b.Y1), int(b.X2), int(b.Y2)).Canon()
}

// Intersection calculates the intersection area between two bounding boxes.
//
// Arguments:
// - other: The other bounding box to calculate intersection with.
//
// Returns:
// - The area of intersection in pixels as float32.
//
// @example
// box1 := BoundingBox{x1: 0, y1: 0, x2: 100, y2: 100}
// box2 := BoundingBox{x1: 50, y1: 50, x2: 150, y2: 150}
// area := box1.Intersection(&box2) // Returns 2500.0 (50x50 overlap)
func (b *BoundingBox) Intersection(other *BoundingBox) float32 {
	r1 := b.ToRect()
	r2 := other.ToRect()
	intersected := r1.Intersect(r2).Canon().Size()
	return float32(intersected.X * intersected.Y)
}

// Union calculates the union area between two bounding boxes.
//
// Arguments:
// - other: The other bounding box to calculate union with.
//
// Returns:
// - The area of union in pixels as float32.
//
// @example
// box1 := BoundingBox{x1: 0, y1: 0, x2: 100, y2: 100}
// box2 := BoundingBox{x1: 50, y1: 50, x2: 150, y2: 150}
// area := box1.Union(&box2) // Returns 17500.0
func (b *BoundingBox) Union(other *BoundingBox) float32 {
	intersectArea := b.Intersection(other)
	r1 := b.ToRect()
	r2 := other.ToRect()
	size1 := r1.Size()
	size2 := r2.Size()
	totalArea := float32(size1.X*size1.Y + size2.X*size2.Y)
	return totalArea - intersectArea
}

// IoU calculates the Intersection over Union between two bounding boxes.
//
// This metric is used for Non-Maximum Suppression (NMS) to remove duplicate detections.
//
// Arguments:
// - other: The other bounding box to calculate IoU with.
//
// Returns:
// - The IoU value between 0 and 1.
//
// @example
// box1 := BoundingBox{x1: 0, y1: 0, x2: 100, y2: 100}
// box2 := BoundingBox{x1: 50, y1: 50, x2: 150, y2: 150}
// iou := box1.IoU(&box2) // Returns ~0.143 (2500/17500)
func (b *BoundingBox) IoU(other *BoundingBox) float32 {
	return b.Intersection(other) / b.Union(other)
}

// ExtractMultiScaleFeatures extracts features at multiple scales from an input image.
//
// This function simulates a Feature Pyramid Network (FPN) backbone by creating
// downsampled feature maps at different scales. In a real implementation, these
// would come from a CNN backbone like ResNet.
//
// Arguments:
// - img: The input image to extract features from.
// - session: The D-FINE model session containing feature specifications.
//
// Returns:
// - An error if feature extraction fails, nil otherwise.
//
// @example
// pic, _ := loadImageFile("image.jpg")
// session := &DFINEModelSession{FeatStrides: []int{8, 16, 32}, ...}
// err := ExtractMultiScaleFeatures(pic, session)
func ExtractMultiScaleFeatures(img image.Image, strides []int, channel int, data []float32) error {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()

	// For each feature level, create appropriate feature maps
	for i, stride := range strides {
		featWidth := width / stride
		featHeight := height / stride

		// Get tensor data slice
		expectedSize := channel * featHeight * featWidth
		if len(data) < expectedSize {
			return fmt.Errorf("Feature tensor %d too small: has %d, needs %d",
				i, len(data), expectedSize)
		}

		// In a real implementation, this would be CNN features
		// For now, we'll create a simplified representation
		resized := resize.Resize(uint(featWidth), uint(featHeight), img, resize.Lanczos3)

		// Fill the tensor with normalized pixel values
		// This is a placeholder - real features would come from a backbone network
		idx := 0
		for c := 0; c < channel; c++ {
			for y := 0; y < featHeight; y++ {
				for x := 0; x < featWidth; x++ {
					if c < 3 { // Use RGB channels if available
						r, g, b, _ := resized.At(x, y).RGBA()
						switch c {
						case 0:
							data[idx] = float32(r>>8) / 255.0
						case 1:
							data[idx] = float32(g>>8) / 255.0
						case 2:
							data[idx] = float32(b>>8) / 255.0
						}
					} else {
						// Fill other channels with learned features (placeholder)
						data[idx] = 0.0
					}
					idx++
				}
			}
		}
	}

	return nil
}

// Package images - This file contains the commonly used utility functions.
package images

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"image"
	"image/jpeg"

	"gocv.io/x/gocv"
)

// CalculateIoU calculates the Intersection over Union between two rectangles.
//
// Arguments:
//   - box1: The first rectangle.
//   - box2: The second rectangle.
//
// Returns:
//   - float32: The Intersection over Union value.
//
// Example:
//
// ```go
//
//	box1 := image.Rect(0, 0, 100, 100)
//	box2 := image.Rect(50, 50, 150, 150)
//	iou := CalculateIoU(box1, box2)
//	fmt.Printf("IoU: %f\n", iou) // IoU: 0.14285714
//
// ```
func CalculateIoU(box1, box2 image.Rectangle) float32 {
	// Calculate intersection
	x1 := max(box1.Min.X, box2.Min.X)
	y1 := max(box1.Min.Y, box2.Min.Y)
	x2 := min(box1.Max.X, box2.Max.X)
	y2 := min(box1.Max.Y, box2.Max.Y)

	if x2 <= x1 || y2 <= y1 {
		return 0.0
	}

	intersection := (x2 - x1) * (y2 - y1)

	// Calculate union
	area1 := (box1.Max.X - box1.Min.X) * (box1.Max.Y - box1.Min.Y)
	area2 := (box2.Max.X - box2.Min.X) * (box2.Max.Y - box2.Min.Y)
	union := area1 + area2 - intersection

	return float32(intersection) / float32(union)
}

// ComputeMatChecksum generates a deterministic checksum for a Mat to verify idempotency.
//
// Arguments:
// - mat: The Mat to compute checksum for.
//
// Returns:
// - A hex-encoded MD5 checksum string.
//
// Example:
//
// ```go
//
//	checksum := ComputeMatChecksum(frame)
//	fmt.Printf("Frame checksum: %s\n", checksum)
//
// ```
func ComputeMatChecksum(mat gocv.Mat) string {
	if mat.Empty() {
		return "empty"
	}

	data, _ := mat.DataPtrUint8()
	hash := md5.New()
	hash.Write(data)
	return fmt.Sprintf("%x", hash.Sum(nil))
}

// DecodeJPEGBytes decodes a JPEG image from a byte slice and returns an image.Image.
//
// Arguments:
//   - data: The JPEG image data as a byte slice.
//
// Returns:
//   - image.Image: The decoded image.
//   - error: An error if decoding fails.
func DecodeJPEGBytes(data []byte) (image.Image, error) {
	reader := bytes.NewReader(data)
	img, err := jpeg.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode JPEG: %w", err)
	}
	return img, nil
}

// DecodeJPEG decodes a JPEG byte slice into an image.Image with minimal overhead.
// It avoids unnecessary allocations and uses a zero-copy reader.
func DecodeJPEG(data []byte) (image.Image, error) {
	return jpeg.Decode(bytes.NewReader(data))
}

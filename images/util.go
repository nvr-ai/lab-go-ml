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

// ApplyNMS applies Non-Maximum Suppression to remove overlapping bounding boxes.
//
// Arguments:
//   - detections: The detections to apply NMS to.
//   - nms: The Non-Maximum Suppression threshold.
//
// Returns:
//   - The detections after applying NMS.
func ApplyNMS(detections []Rect, nms float32) []Rect {
	if len(detections) == 0 {
		return detections
	}

	var result []Rect
	used := make([]bool, len(detections))

	for i := 0; i < len(detections); i++ {
		if used[i] {
			continue
		}

		result = append(result, detections[i])
		used[i] = true

		// Check overlap with remaining detections
		for j := i + 1; j < len(detections); j++ {
			if used[j] {
				continue
			}

			// Calculate IoU
			iou := CalculateIoU(
				detections[i].X1,
				detections[i].Y1,
				detections[i].X2,
				detections[i].Y2,
				detections[j].X1,
				detections[j].Y1,
				detections[j].X2,
				detections[j].Y2,
			)
			if iou > nms {
				used[j] = true
			}
		}
	}

	return result
}

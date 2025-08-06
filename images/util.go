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

package images

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"

	"github.com/chai2010/webp"
	"github.com/cshum/vipsgen/vips"
	"gocv.io/x/gocv"
)

// ResizeJPEG resizes a JPEG []byte to the given width and height, returning a gocv.Mat.
//
// Arguments:
//   - jpegBytes: The JPEG []byte to resize.
//   - width: The width to resize the image to.
//   - height: The height to resize the image to.
//
// Returns:
//   - gocv.Mat: The resized image.
//   - error: An error if the image fails to resize.
func ResizeJPEG(jpegBytes []byte, width, height int) (gocv.Mat, error) {
	// Load the image from buffer.
	img, err := vips.NewImageFromBuffer(jpegBytes, &vips.LoadOptions{
		Access: vips.AccessSequential,
	})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to load image: %w", err)
	}
	defer img.Close()

	// Resize the image in-place.
	err = img.ThumbnailImage(width, &vips.ThumbnailImageOptions{
		Height: height,
		FailOn: vips.FailOnError,
	})
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("failed to resize image: %w", err)
	}

	// Export to JPEG buffer.
	resized, err := img.JpegsaveBuffer(&vips.JpegsaveBufferOptions{})
	if err != nil || len(resized) == 0 {
		return gocv.NewMat(), fmt.Errorf("failed to encode resized image")
	}

	// Decode into gocv.Mat so we can use it with OpenCV.
	mat, err := gocv.IMDecode(resized, gocv.IMReadColor)
	if err != nil || mat.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to decode resized image")
	}

	return mat, nil
}

// ResizeJPEGToImage resizes a JPEG []byte to the given width and height,
// returning a Go-native image.Image.
func ResizeJPEGToImage(jpegBytes []byte, width, height int) (image.Image, error) {
	// Load the image from buffer
	img, err := vips.NewImageFromBuffer(jpegBytes, &vips.LoadOptions{
		Access: vips.AccessSequential,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load image: %w", err)
	}
	defer img.Close()

	// Resize the image in-place
	err = img.ThumbnailImage(width, &vips.ThumbnailImageOptions{
		Height: height,
		FailOn: vips.FailOnError,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to resize image: %w", err)
	}

	// Export to JPEG buffer
	resizedBytes, err := img.JpegsaveBuffer(&vips.JpegsaveBufferOptions{})
	if err != nil || len(resizedBytes) == 0 {
		return nil, fmt.Errorf("failed to encode resized image")
	}

	// Decode into image.Image
	imgDecoded, err := jpeg.Decode(bytes.NewReader(resizedBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to decode resized JPEG: %w", err)
	}

	return imgDecoded, nil
}

// ResizeWebPToImage resizes a WebP []byte to the given width and height,
// returning a Go-native image.Image.
func ResizeWebPToImage(b []byte, width, height int) (image.Image, error) {
	// Load the image from buffer
	img, err := vips.NewImageFromBuffer(b, &vips.LoadOptions{
		Access: vips.AccessSequential,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load image: %w", err)
	}
	defer img.Close()

	// Resize the image in-place
	err = img.ThumbnailImage(width, &vips.ThumbnailImageOptions{
		Height: height,
		FailOn: vips.FailOnError,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to resize image: %w", err)
	}

	// Export to JPEG buffer
	resizedBytes, err := img.WebpsaveBuffer(&vips.WebpsaveBufferOptions{})
	if err != nil || len(resizedBytes) == 0 {
		return nil, fmt.Errorf("failed to encode resized image")
	}

	// Decode into image.Image
	imgDecoded, err := webp.Decode(bytes.NewReader(resizedBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to decode resized WebP: %w", err)
	}

	return imgDecoded, nil
}

// ImageFormat represents supported image formats
type ImageFormat int

const (
	FormatJPEG ImageFormat = iota
	FormatWebP
	FormatPNG
)

// ResizePNGToImage resizes a PNG []byte to the given width and height,
// returning a Go-native image.Image.
func ResizePNGToImage(b []byte, width, height int) (image.Image, error) {
	// Load the image from buffer
	img, err := vips.NewImageFromBuffer(b, &vips.LoadOptions{
		Access: vips.AccessSequential,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load image: %w", err)
	}
	defer img.Close()

	// Resize the image in-place
	err = img.ThumbnailImage(width, &vips.ThumbnailImageOptions{
		Height: height,
		FailOn: vips.FailOnError,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to resize image: %w", err)
	}

	// Export to PNG buffer
	resizedBytes, err := img.PngsaveBuffer(&vips.PngsaveBufferOptions{})
	if err != nil || len(resizedBytes) == 0 {
		return nil, fmt.Errorf("failed to encode resized image")
	}

	// Decode into image.Image
	imgDecoded, err := png.Decode(bytes.NewReader(resizedBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to decode resized PNG: %w", err)
	}

	return imgDecoded, nil
}

// ResizeImageToImage provides a unified interface to resize images of different formats
// to image.Image, suitable for ONNX runtime inference.
func ResizeImageToImage(imageBytes []byte, width, height int, format ImageFormat) (image.Image, error) {
	if len(imageBytes) == 0 {
		return nil, fmt.Errorf("empty image data")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions: width=%d, height=%d", width, height)
	}

	switch format {
	case FormatJPEG:
		return ResizeJPEGToImage(imageBytes, width, height)
	case FormatWebP:
		return ResizeWebPToImage(imageBytes, width, height)
	case FormatPNG:
		return ResizePNGToImage(imageBytes, width, height)
	default:
		return nil, fmt.Errorf("unsupported image format: %d", format)
	}
}

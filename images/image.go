// Package images - Image definition for processing utilities.
package images

// Image represents an image with a format, data, width, and height.
type Image struct {
	// The format of the image.
	Format ImageFormat `json:"format" yaml:"format"`
	// The data of the image.
	Data []byte `json:"data" yaml:"data"`
	// The width of the image.
	Width int `json:"width" yaml:"width"`
	// The height of the image.
	Height int `json:"height" yaml:"height"`
}

// ImageFormat represents supported image formats
type ImageFormat string

// ImageFormat constants
const (
	// FormatJPEG is the JPEG image format.
	FormatJPEG ImageFormat = "jpeg"
	// FormatWebP is the WebP image format.
	FormatWebP ImageFormat = "webp"
	// FormatPNG is the PNG image format.
	FormatPNG ImageFormat = "png"
)

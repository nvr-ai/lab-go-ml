package images

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"testing"

	"github.com/chai2010/webp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func getTestImage() image.Image {
	// Create a simple 100x100 red image and encode as JPEG.
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: 255, G: 0, B: 0, A: 255})
		}
	}

	return img
}

// TestResizeJPEG validates the ResizeJPEG function for resizing and error cases.
func TestResizeJPEG(t *testing.T) {
	var buf bytes.Buffer
	err := jpeg.Encode(&buf, getTestImage(), nil)
	assert.NoError(t, err, "JPEG encoding should succeed")

	jpegBytes := buf.Bytes()
	targetWidth, targetHeight := 50, 50

	// Test successful resize
	mat, err := ResizeJPEG(jpegBytes, targetWidth, targetHeight)
	assert.NoError(t, err, "ResizeJPEG should not error for valid input")
	assert.False(t, mat.Empty(), "Resulting Mat should not be empty")
	assert.Equal(t, targetHeight, mat.Rows(), "Mat should have correct height")
	assert.Equal(t, targetWidth, mat.Cols(), "Mat should have correct width")
	mat.Close()

	// Test with invalid JPEG data
	badBytes := []byte("not a jpeg")
	mat, err = ResizeJPEG(badBytes, targetWidth, targetHeight)
	assert.Error(t, err, "ResizeJPEG should error for invalid JPEG input")
	assert.True(t, mat.Empty(), "Mat should be empty on error")
	mat.Close()

	// Test with zero dimensions
	mat, err = ResizeJPEG(jpegBytes, 0, 0)
	assert.Error(t, err, "ResizeJPEG should error for zero dimensions")
	assert.True(t, mat.Empty(), "Mat should be empty on error")
	mat.Close()
}

func TestResizeWebP(t *testing.T) {
	buf := bytes.Buffer{}
	err := jpeg.Encode(&buf, getTestImage(), nil)
	assert.NoError(t, err, "JPEG encoding should succeed")

	img, err := ResizeWebPToImage(buf.Bytes(), 240, 240)
	assert.NoError(t, err, "ResizeWebP should not error for valid input")
	assert.NotNil(t, img, "Resulting image should not be nil")
	assert.Equal(t, 240, img.Bounds().Canon().Dx(), "Mat should have correct width")
	assert.Equal(t, 240, img.Bounds().Canon().Dy(), "Mat should have correct height")
}

// Helper functions to create test data for different formats
func getJPEGBytes(t *testing.T) []byte {
	var buf bytes.Buffer
	err := jpeg.Encode(&buf, getTestImage(), nil)
	require.NoError(t, err)
	return buf.Bytes()
}

func getPNGBytes(t *testing.T) []byte {
	var buf bytes.Buffer
	err := png.Encode(&buf, getTestImage())
	require.NoError(t, err)
	return buf.Bytes()
}

func getWebPBytes(t *testing.T) []byte {
	var buf bytes.Buffer
	err := webp.Encode(&buf, getTestImage(), &webp.Options{Quality: 80})
	require.NoError(t, err)
	return buf.Bytes()
}

// Test PNG resizing functionality
func TestResizePNGToImage(t *testing.T) {
	pngBytes := getPNGBytes(t)
	targetWidth, targetHeight := 50, 50

	// Test successful resize
	img, err := ResizePNGToImage(pngBytes, targetWidth, targetHeight)
	assert.NoError(t, err, "ResizePNGToImage should not error for valid input")
	assert.NotNil(t, img, "Resulting image should not be nil")
	assert.Equal(t, targetWidth, img.Bounds().Dx(), "Image should have correct width")
	assert.Equal(t, targetHeight, img.Bounds().Dy(), "Image should have correct height")

	// Test with invalid PNG data
	badBytes := []byte("not a png")
	img, err = ResizePNGToImage(badBytes, targetWidth, targetHeight)
	assert.Error(t, err, "ResizePNGToImage should error for invalid PNG input")
	assert.Nil(t, img, "Image should be nil on error")

	// Test with zero dimensions
	img, err = ResizePNGToImage(pngBytes, 0, 0)
	assert.Error(t, err, "ResizePNGToImage should error for zero dimensions")
	assert.Nil(t, img, "Image should be nil on error")
}

// Test the unified ResizeImageToImage interface
func TestResizeImageToImage(t *testing.T) {
	tests := []struct {
		name       string
		format     ImageFormat
		getBytes   func(t *testing.T) []byte
		targetW    int
		targetH    int
		shouldFail bool
	}{
		{
			name:     "JPEG resize success",
			format:   FormatJPEG,
			getBytes: getJPEGBytes,
			targetW:  64, targetH: 64,
			shouldFail: false,
		},
		{
			name:     "WebP resize success",
			format:   FormatWebP,
			getBytes: getWebPBytes,
			targetW:  128, targetH: 128,
			shouldFail: false,
		},
		{
			name:     "PNG resize success",
			format:   FormatPNG,
			getBytes: getPNGBytes,
			targetW:  32, targetH: 32,
			shouldFail: false,
		},
		{
			name:     "Invalid dimensions",
			format:   FormatJPEG,
			getBytes: getJPEGBytes,
			targetW:  0, targetH: 0,
			shouldFail: true,
		},
		{
			name:     "Negative dimensions",
			format:   FormatJPEG,
			getBytes: getJPEGBytes,
			targetW:  -10, targetH: 50,
			shouldFail: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			imageBytes := tt.getBytes(t)

			img, err := ResizeImageToImage(imageBytes, tt.targetW, tt.targetH, tt.format)

			if tt.shouldFail {
				assert.Error(t, err)
				assert.Nil(t, img)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, img)
				assert.Equal(t, tt.targetW, img.Bounds().Dx())
				assert.Equal(t, tt.targetH, img.Bounds().Dy())
			}
		})
	}
}

// Test edge cases for the unified interface
func TestResizeImageToImageEdgeCases(t *testing.T) {
	jpegBytes := getJPEGBytes(t)

	// Test empty image data
	img, err := ResizeImageToImage([]byte{}, 50, 50, FormatJPEG)
	assert.Error(t, err, "Should error with empty image data")
	assert.Nil(t, img)
	assert.Contains(t, err.Error(), "empty image data")

	// Test unsupported format
	img, err = ResizeImageToImage(jpegBytes, 50, 50, FormatJPEG)
	assert.Error(t, err, "Should error with unsupported format")
	assert.Nil(t, img)
	assert.Contains(t, err.Error(), "unsupported image format")
}

// Benchmark tests for performance comparison between formats
func BenchmarkResizeJPEG(b *testing.B) {
	jpegBytes := getJPEGBytes(&testing.T{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		img, err := ResizeJPEGToImage(jpegBytes, 224, 224)
		if err != nil {
			b.Fatal(err)
		}
		_ = img
	}
}

func BenchmarkResizeWebP(b *testing.B) {
	webpBytes := getWebPBytes(&testing.T{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		img, err := ResizeWebPToImage(webpBytes, 224, 224)
		if err != nil {
			b.Fatal(err)
		}
		_ = img
	}
}

func BenchmarkResizePNG(b *testing.B) {
	pngBytes := getPNGBytes(&testing.T{})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		img, err := ResizePNGToImage(pngBytes, 224, 224)
		if err != nil {
			b.Fatal(err)
		}
		_ = img
	}
}

// Benchmark the unified interface
func BenchmarkResizeImageToImage(b *testing.B) {
	tests := []struct {
		name     string
		format   ImageFormat
		getBytes func(*testing.T) []byte
	}{
		{"JPEG", FormatJPEG, getJPEGBytes},
		{"WebP", FormatWebP, getWebPBytes},
		{"PNG", FormatPNG, getPNGBytes},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			imageBytes := tt.getBytes(&testing.T{})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				img, err := ResizeImageToImage(imageBytes, 224, 224, tt.format)
				if err != nil {
					b.Fatal(err)
				}
				_ = img
			}
		})
	}
}

// Benchmark different target sizes to understand scaling performance
func BenchmarkResizeDifferentSizes(b *testing.B) {
	jpegBytes := getJPEGBytes(&testing.T{})
	sizes := []struct {
		name string
		w, h int
	}{
		{"Small_64x64", 64, 64},
		{"Medium_224x224", 224, 224},
		{"Large_512x512", 512, 512},
		{"VeryLarge_1024x1024", 1024, 1024},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				img, err := ResizeImageToImage(jpegBytes, size.w, size.h, FormatJPEG)
				if err != nil {
					b.Fatal(err)
				}
				_ = img
			}
		})
	}
}

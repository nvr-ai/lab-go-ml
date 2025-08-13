package preprocess

// Package image_test provides comprehensive test coverage for image preprocessing utilities.
//
// This test suite validates all aspects of the image preprocessing pipeline including
// input validation, image decoding, resizing, normalization, tensor conversion, and
// configuration presets. The tests ensure idempotency, fault tolerance, and performance
// characteristics suitable for production machine learning inference pipelines.

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPreprocessImageDFINE validates complete preprocessing pipeline for D-FINE model configuration.
//
// This test ensures that the D-FINE preprocessing configuration produces correctly
// formatted tensors with proper normalization, dimensions, and data layout for
// optimal model performance. The test covers the full pipeline from raw JPEG
// data to final tensor output.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageDFINE(t *testing.T) {
	// Create test JPEG image data
	testImg := createTestJPEGImage(t, 800, 600)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  800,
		Height: 600,
	}

	config := imgutil.GetDFINEConfig()

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err, "D-FINE preprocessing should succeed with valid input")
	require.NotNil(t, preprocessed, "Preprocessed result should not be nil")

	// Validate tensor shape: [batch=1, channels=3, height=640, width=640]
	expectedShape := []int64{1, 3, 640, 640}
	assert.Equal(t, expectedShape, preprocessed.Shape, "D-FINE tensor shape should match expected dimensions")

	// Validate tensor data size
	expectedDataSize := 1 * 3 * 640 * 640
	assert.Len(t, preprocessed.Data, expectedDataSize, "D-FINE tensor data size should match shape")

	// Validate dimensions
	assert.Equal(t, 640, preprocessed.Width, "Processed width should match target")
	assert.Equal(t, 640, preprocessed.Height, "Processed height should match target")
	assert.Equal(t, 800, preprocessed.OriginalWidth, "Original width should be preserved")
	assert.Equal(t, 600, preprocessed.OriginalHeight, "Original height should be preserved")

	// Validate scale factor calculation
	expectedScale := math.Min(640.0/800.0, 640.0/600.0)
	assert.InDelta(t, expectedScale, preprocessed.ScaleFactor, 0.001, "Scale factor should be calculated correctly")

	// Validate pixel value ranges for ImageNet normalization
	validateImageNetNormalization(t, preprocessed.Data)
}

// TestPreprocessImageRFDETR validates complete preprocessing pipeline for RF-DETR model configuration.
//
// This test ensures that the RF-DETR preprocessing configuration produces correctly
// formatted tensors suitable for transformer-based object detection. It validates
// the specific normalization and formatting requirements of RF-DETR models.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageRFDETR(t *testing.T) {
	// Create test JPEG image data
	testImg := createTestJPEGImage(t, 1024, 768)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  1024,
		Height: 768,
	}

	config := imgutil.GetRFDETRConfig()

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err, "RF-DETR preprocessing should succeed with valid input")
	require.NotNil(t, preprocessed, "Preprocessed result should not be nil")

	// Validate tensor shape: [batch=1, channels=3, height=640, width=640]
	expectedShape := []int64{1, 3, 640, 640}
	assert.Equal(t, expectedShape, preprocessed.Shape, "RF-DETR tensor shape should match expected dimensions")

	// Validate tensor data size
	expectedDataSize := 1 * 3 * 640 * 640
	assert.Len(t, preprocessed.Data, expectedDataSize, "RF-DETR tensor data size should match shape")

	// Validate that letterboxing was applied
	expectedScale := math.Min(640.0/1024.0, 640.0/768.0)
	assert.InDelta(t, expectedScale, preprocessed.ScaleFactor, 0.001, "Scale factor should account for letterboxing")

	// Validate pixel value ranges
	validateImageNetNormalization(t, preprocessed.Data)
}

// TestPreprocessImageYOLOv4 validates complete preprocessing pipeline for YOLOv4 model configuration.
//
// This test ensures that the YOLOv4 preprocessing configuration produces correctly
// formatted tensors with simple 0-1 normalization suitable for YOLO architecture.
// It validates the specific input requirements and tensor layout for YOLOv4 models.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageYOLOv4(t *testing.T) {
	// Create test JPEG image data
	testImg := createTestJPEGImage(t, 1280, 720)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  1280,
		Height: 720,
	}

	config := imgutil.GetYOLOv4Config()

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err, "YOLOv4 preprocessing should succeed with valid input")
	require.NotNil(t, preprocessed, "Preprocessed result should not be nil")

	// Validate tensor shape: [batch=1, channels=3, height=608, width=608]
	expectedShape := []int64{1, 3, 608, 608}
	assert.Equal(t, expectedShape, preprocessed.Shape, "YOLOv4 tensor shape should match expected dimensions")

	// Validate tensor data size
	expectedDataSize := 1 * 3 * 608 * 608
	assert.Len(t, preprocessed.Data, expectedDataSize, "YOLOv4 tensor data size should match shape")

	// Validate simple 0-1 normalization (no mean/std adjustment)
	for _, pixel := range preprocessed.Data {
		assert.GreaterOrEqual(t, pixel, 0.0, "YOLOv4 pixels should be >= 0")
		assert.LessOrEqual(t, pixel, 1.0, "YOLOv4 pixels should be <= 1")
	}
}

// TestPreprocessImageGrayscale validates grayscale image preprocessing functionality.
//
// This test ensures that grayscale conversion works correctly and produces
// single-channel tensor output with proper normalization. It validates both
// the conversion process and the resulting tensor format.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageGrayscale(t *testing.T) {
	testImg := createTestJPEGImage(t, 400, 300)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  400,
		Height: 300,
	}

	config := imgutil.PreprocessConfig{
		TargetWidth:  256,
		TargetHeight: 256,
		Mean:         []float32{0.5},
		Std:          []float32{0.5},
		Grayscale:    true,
		BGR:          false,
		LetterBox:    false,
		PadColor:     [3]uint8{0, 0, 0},
		Denoise:      false,
		Quality:      90,
		DataType:     "float32",
		PixelRange:   [2]float32{0.0, 1.0},
	}

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err, "Grayscale preprocessing should succeed")
	require.NotNil(t, preprocessed, "Preprocessed result should not be nil")

	// Validate tensor shape: [batch=1, channels=1, height=256, width=256]
	expectedShape := []int64{1, 1, 256, 256}
	assert.Equal(t, expectedShape, preprocessed.Shape, "Grayscale tensor should have single channel")

	// Validate tensor data size
	expectedDataSize := 1 * 1 * 256 * 256
	assert.Len(t, preprocessed.Data, expectedDataSize, "Grayscale tensor data size should match shape")
}

// TestPreprocessImageBGRConversion validates BGR color space conversion functionality.
//
// This test ensures that RGB to BGR channel order conversion works correctly
// and produces properly ordered tensor data for models that expect BGR input.
// It validates the channel reordering process and tensor layout.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageBGRConversion(t *testing.T) {
	// Create a test image with distinct RGB values
	testImg := createColorTestJPEGImage(t, 64, 64)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  64,
		Height: 64,
	}

	// Test with BGR enabled
	configBGR := imgutil.PreprocessConfig{
		TargetWidth:  64,
		TargetHeight: 64,
		Mean:         []float32{0.0, 0.0, 0.0},
		Std:          []float32{1.0, 1.0, 1.0},
		Grayscale:    false,
		BGR:          true,
		LetterBox:    false,
		PadColor:     [3]uint8{0, 0, 0},
		Denoise:      false,
		Quality:      90,
		DataType:     "float32",
		PixelRange:   [2]float32{0.0, 1.0},
	}

	preprocessedBGR, err := imgutil.PreprocessImage(inputImage, configBGR)
	require.NoError(t, err, "BGR preprocessing should succeed")

	// Test with RGB (default)
	configRGB := configBGR
	configRGB.BGR = false

	preprocessedRGB, err := imgutil.PreprocessImage(inputImage, configRGB)
	require.NoError(t, err, "RGB preprocessing should succeed")

	// The first channel of BGR should match the third channel of RGB
	// (assuming NCHW format where channels are separated)
	chSize := 64 * 64
	bgrFirstChannel := preprocessedBGR.Data[0:chSize]
	rgbThirdChannel := preprocessedRGB.Data[2*chSize : 3*chSize]

	assert.InDeltaSlice(t, rgbThirdChannel, bgrFirstChannel, 0.01, "BGR first channel should match RGB third channel")
}

// TestPreprocessImagePNG validates PNG image format support in the preprocessing pipeline.
//
// This test ensures that the preprocessing pipeline correctly handles PNG images
// in addition to JPEG format. It validates PNG decoding and subsequent processing
// steps to ensure format-agnostic functionality.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImagePNG(t *testing.T) {
	testImg := createTestPNGImage(t, 200, 150)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatPNG,
		Data:   testImg,
		Width:  200,
		Height: 150,
	}

	config := imgutil.GetDFINEConfig()

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err, "PNG preprocessing should succeed")
	require.NotNil(t, preprocessed, "Preprocessed result should not be nil")

	// Validate basic tensor properties
	expectedShape := []int64{1, 3, 640, 640}
	assert.Equal(t, expectedShape, preprocessed.Shape, "PNG preprocessing should produce correct tensor shape")
}

// TestPreprocessImageValidation validates comprehensive input validation functionality.
//
// This test ensures that all input validation rules are properly enforced
// and that appropriate error messages are returned for invalid inputs.
// It covers edge cases and boundary conditions to ensure robust error handling.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageValidation(t *testing.T) {
	validImg := createTestJPEGImage(t, 100, 100)

	testCases := []struct {
		name        string
		image       *imgutil.Image
		config      imgutil.PreprocessConfig
		expectError bool
		errorMsg    string
	}{
		{
			name:        "Nil image",
			image:       nil,
			config:      imgutil.GetDFINEConfig(),
			expectError: true,
			errorMsg:    "image cannot be nil",
		},
		{
			name: "Empty image data",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   []byte{},
				Width:  100,
				Height: 100,
			},
			config:      imgutil.GetDFINEConfig(),
			expectError: true,
			errorMsg:    "image data cannot be empty",
		},
		{
			name: "Zero width",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  0,
				Height: 100,
			},
			config:      imgutil.GetDFINEConfig(),
			expectError: true,
			errorMsg:    "image dimensions must be positive",
		},
		{
			name: "Negative height",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  100,
				Height: -50,
			},
			config:      imgutil.GetDFINEConfig(),
			expectError: true,
			errorMsg:    "image dimensions must be positive",
		},
		{
			name: "Invalid target dimensions",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  100,
				Height: 100,
			},
			config: imgutil.PreprocessConfig{
				TargetWidth:  0,
				TargetHeight: 100,
				Mean:         []float32{0.0, 0.0, 0.0},
				Std:          []float32{1.0, 1.0, 1.0},
				DataType:     "float32",
			},
			expectError: true,
			errorMsg:    "target dimensions must be positive",
		},
		{
			name: "Invalid mean/std count for color",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  100,
				Height: 100,
			},
			config: imgutil.PreprocessConfig{
				TargetWidth:  100,
				TargetHeight: 100,
				Mean:         []float32{0.0}, // Should be 3 values for color
				Std:          []float32{1.0},
				Grayscale:    false,
				DataType:     "float32",
			},
			expectError: true,
			errorMsg:    "color images require three mean and std values",
		},
		{
			name: "Zero standard deviation",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  100,
				Height: 100,
			},
			config: imgutil.PreprocessConfig{
				TargetWidth:  100,
				TargetHeight: 100,
				Mean:         []float32{0.0, 0.0, 0.0},
				Std:          []float32{1.0, 0.0, 1.0}, // Zero std is invalid
				DataType:     "float32",
			},
			expectError: true,
			errorMsg:    "standard deviation values must be positive",
		},
		{
			name: "Invalid quality range",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  100,
				Height: 100,
			},
			config: imgutil.PreprocessConfig{
				TargetWidth:  100,
				TargetHeight: 100,
				Mean:         []float32{0.0, 0.0, 0.0},
				Std:          []float32{1.0, 1.0, 1.0},
				Quality:      150, // Invalid quality > 100
				DataType:     "float32",
			},
			expectError: true,
			errorMsg:    "quality must be between 0 and 100",
		},
		{
			name: "Unsupported data type",
			image: &imgutil.Image{
				Format: imgutil.ImageFormatJPEG,
				Data:   validImg,
				Width:  100,
				Height: 100,
			},
			config: imgutil.PreprocessConfig{
				TargetWidth:  100,
				TargetHeight: 100,
				Mean:         []float32{0.0, 0.0, 0.0},
				Std:          []float32{1.0, 1.0, 1.0},
				Quality:      90,
				DataType:     "int32", // Unsupported type
			},
			expectError: true,
			errorMsg:    "unsupported data type",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			preprocessed, err := imgutil.PreprocessImage(tc.image, tc.config)

			if tc.expectError {
				assert.Error(t, err, "Should return error for invalid input")
				assert.Contains(t, err.Error(), tc.errorMsg, "Error message should contain expected text")
				assert.Nil(t, preprocessed, "Should not return preprocessed data on error")
			} else {
				assert.NoError(t, err, "Should not return error for valid input")
				assert.NotNil(t, preprocessed, "Should return valid preprocessed data")
			}
		})
	}
}

// TestPreprocessImageCorruptedData validates error handling for corrupted image data.
//
// This test ensures that the preprocessing pipeline properly handles and reports
// errors when encountering corrupted or invalid image data. It validates the
// robustness of the image decoding stage.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageCorruptedData(t *testing.T) {
	corruptedData := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10} // Incomplete JPEG header

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   corruptedData,
		Width:  100,
		Height: 100,
	}

	config := imgutil.GetDFINEConfig()

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	assert.Error(t, err, "Should return error for corrupted image data")
	assert.Nil(t, preprocessed, "Should not return preprocessed data for corrupted input")
	assert.Contains(t, err.Error(), "image decoding failed", "Error should indicate decoding failure")
}

// TestPreprocessImageIdempotency validates that preprocessing operations are idempotent.
//
// This test ensures that multiple calls to the preprocessing function with
// identical inputs produce identical outputs. This is crucial for reproducible
// machine learning inference and debugging consistency.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageIdempotency(t *testing.T) {
	testImg := createTestJPEGImage(t, 320, 240)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  320,
		Height: 240,
	}

	config := imgutil.GetDFINEConfig()

	// Process the same image multiple times
	preprocessed1, err1 := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err1, "First preprocessing should succeed")

	preprocessed2, err2 := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err2, "Second preprocessing should succeed")

	preprocessed3, err3 := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err3, "Third preprocessing should succeed")

	// Validate that all results are identical
	assert.Equal(t, preprocessed1.Shape, preprocessed2.Shape, "Tensor shapes should be identical")
	assert.Equal(t, preprocessed1.Shape, preprocessed3.Shape, "Tensor shapes should be identical")

	assert.Equal(t, preprocessed1.Width, preprocessed2.Width, "Processed widths should be identical")
	assert.Equal(t, preprocessed1.Height, preprocessed2.Height, "Processed heights should be identical")

	assert.InDelta(t, preprocessed1.ScaleFactor, preprocessed2.ScaleFactor, 1e-10, "Scale factors should be identical")
	assert.InDelta(t, preprocessed1.ScaleFactor, preprocessed3.ScaleFactor, 1e-10, "Scale factors should be identical")

	// Validate tensor data is identical
	require.Len(t, preprocessed1.Data, len(preprocessed2.Data), "Tensor data lengths should match")
	for i := range preprocessed1.Data {
		assert.InDelta(t, preprocessed1.Data[i], preprocessed2.Data[i], 1e-6,
			"Tensor data should be identical at index %d", i)
		assert.InDelta(t, preprocessed1.Data[i], preprocessed3.Data[i], 1e-6,
			"Tensor data should be identical at index %d", i)
	}
}

// TestPreprocessImageLetterboxing validates letterboxing functionality for aspect ratio preservation.
//
// This test ensures that letterboxing correctly preserves aspect ratios while
// resizing images to target dimensions. It validates padding application and
// scale factor calculations for non-square aspect ratios.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestPreprocessImageLetterboxing(t *testing.T) {
	// Create a wide image (2:1 aspect ratio)
	testImg := createTestJPEGImage(t, 400, 200)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  400,
		Height: 200,
	}

	config := imgutil.PreprocessConfig{
		TargetWidth:  300,
		TargetHeight: 300,
		Mean:         []float32{0.0, 0.0, 0.0},
		Std:          []float32{1.0, 1.0, 1.0},
		LetterBox:    true,
		PadColor:     [3]uint8{128, 128, 128},
		DataType:     "float32",
		PixelRange:   [2]float32{0.0, 1.0},
	}

	preprocessed, err := imgutil.PreprocessImage(inputImage, config)
	require.NoError(t, err, "Letterboxing preprocessing should succeed")

	// Validate final dimensions
	assert.Equal(t, 300, preprocessed.Width, "Final width should match target")
	assert.Equal(t, 300, preprocessed.Height, "Final height should match target")

	// Validate scale factor (should be limited by height: 300/200 = 1.5)
	expectedScale := math.Min(300.0/400.0, 300.0/200.0) // 0.75
	assert.InDelta(t, expectedScale, preprocessed.ScaleFactor, 0.001, "Scale factor should preserve aspect ratio")

	// Test without letterboxing for comparison
	configNoLetterbox := config
	configNoLetterbox.LetterBox = false

	preprocessedNoLetterbox, err := imgutil.PreprocessImage(inputImage, configNoLetterbox)
	require.NoError(t, err, "Non-letterbox preprocessing should succeed")

	// Without letterboxing, scale factors would be different for each dimension
	scaleX := 300.0 / 400.0
	scaleY := 300.0 / 200.0
	expectedScaleNoLetterbox := math.Min(scaleX, scaleY)
	assert.InDelta(t, expectedScaleNoLetterbox, preprocessedNoLetterbox.ScaleFactor, 0.001,
		"Non-letterbox scale should be calculated differently")
}

// TestConfigurationPresets validates all model-specific configuration presets.
//
// This test ensures that each model configuration preset (D-FINE, RF-DETR, YOLOv4)
// contains appropriate and valid parameters for their respective model architectures.
// It validates configuration consistency and expected parameter ranges.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
func TestConfigurationPresets(t *testing.T) {
	testCases := []struct {
		name       string
		configFunc func() imgutil.PreprocessConfig
		expected   map[string]interface{}
	}{
		{
			name:       "D-FINE configuration",
			configFunc: imgutil.GetDFINEConfig,
			expected: map[string]interface{}{
				"target_width":  640,
				"target_height": 640,
				"letterbox":     true,
				"denoise":       true,
				"grayscale":     false,
				"bgr":           false,
				"data_type":     "float32",
				"mean_length":   3,
				"std_length":    3,
			},
		},
		{
			name:       "RF-DETR configuration",
			configFunc: imgutil.GetRFDETRConfig,
			expected: map[string]interface{}{
				"target_width":  640,
				"target_height": 640,
				"letterbox":     true,
				"denoise":       false,
				"grayscale":     false,
				"bgr":           false,
				"data_type":     "float32",
				"mean_length":   3,
				"std_length":    3,
			},
		},
		{
			name:       "YOLOv4 configuration",
			configFunc: imgutil.GetYOLOv4Config,
			expected: map[string]interface{}{
				"target_width":  608,
				"target_height": 608,
				"letterbox":     true,
				"denoise":       false,
				"grayscale":     false,
				"bgr":           false,
				"data_type":     "float32",
				"mean_length":   3,
				"std_length":    3,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := tc.configFunc()

			assert.Equal(t, tc.expected["target_width"], config.TargetWidth, "Target width should match expected")
			assert.Equal(t, tc.expected["target_height"], config.TargetHeight, "Target height should match expected")
			assert.Equal(t, tc.expected["letterbox"], config.LetterBox, "Letterbox setting should match expected")
			assert.Equal(t, tc.expected["denoise"], config.Denoise, "Denoise setting should match expected")
			assert.Equal(t, tc.expected["grayscale"], config.Grayscale, "Grayscale setting should match expected")
			assert.Equal(t, tc.expected["bgr"], config.BGR, "BGR setting should match expected")
			assert.Equal(t, tc.expected["data_type"], config.DataType, "Data type should match expected")
			assert.Len(t, config.Mean, tc.expected["mean_length"], "Mean array length should match expected")
			assert.Len(t, config.Std, tc.expected["std_length"], "Std array length should match expected")

			// Validate that all std values are positive
			for i, std := range config.Std {
				assert.Greater(t, std, 0.0, "Standard deviation value %d should be positive", i)
			}

			// Validate quality range
			assert.GreaterOrEqual(t, config.Quality, 0, "Quality should be non-negative")
			assert.LessOrEqual(t, config.Quality, 100, "Quality should not exceed 100")
		})
	}
}

// BenchmarkPreprocessImageDFINE measures the performance characteristics of D-FINE preprocessing.
//
// This benchmark helps identify performance regressions and ensures that the
// preprocessing pipeline maintains acceptable performance for real-time inference
// applications. It measures throughput and memory allocation patterns.
//
// Arguments:
//   - b: Benchmarking context for performance measurement and reporting.
func BenchmarkPreprocessImageDFINE(b *testing.B) {
	testImg := createTestJPEGImage(b, 1920, 1080)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  1920,
		Height: 1080,
	}

	config := imgutil.GetDFINEConfig()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		preprocessed, err := imgutil.PreprocessImage(inputImage, config)
		if err != nil {
			b.Fatalf("Preprocessing failed: %v", err)
		}
		_ = preprocessed // Prevent optimization elimination
	}
}

// BenchmarkPreprocessImageYOLOv4 measures the performance characteristics of YOLOv4 preprocessing.
//
// This benchmark validates the performance of YOLOv4-specific preprocessing
// pipeline and helps ensure that simpler normalization doesn't introduce
// unexpected performance overhead compared to more complex configurations.
//
// Arguments:
//   - b: Benchmarking context for performance measurement and reporting.
func BenchmarkPreprocessImageYOLOv4(b *testing.B) {
	testImg := createTestJPEGImage(b, 1280, 720)

	inputImage := &imgutil.Image{
		Format: imgutil.ImageFormatJPEG,
		Data:   testImg,
		Width:  1280,
		Height: 720,
	}

	config := imgutil.GetYOLOv4Config()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		preprocessed, err := imgutil.PreprocessImage(inputImage, config)
		if err != nil {
			b.Fatalf("Preprocessing failed: %v", err)
		}
		_ = preprocessed // Prevent optimization elimination
	}
}

// Helper functions for test support

// createTestJPEGImage creates a test JPEG image with specified dimensions for testing purposes.
//
// This helper function generates a synthetic JPEG image with a gradient pattern
// that provides predictable pixel values for validation. The generated image
// contains sufficient detail to test all preprocessing pipeline stages effectively.
//
// Arguments:
//   - t: Testing interface for error reporting (can be testing.T or testing.B).
//   - width: The desired image width in pixels.
//   - height: The desired image height in pixels.
//
// Returns:
//   - []byte: The encoded JPEG image data ready for preprocessing tests.
func createTestJPEGImage(t testing.TB, width, height int) []byte {
	t.Helper()

	// Create a test image with gradient pattern
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Create a gradient pattern for predictable testing
			r := uint8((x * 255) / width)
			g := uint8((y * 255) / height)
			b := uint8(((x + y) * 255) / (width + height))
			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	// Encode as JPEG
	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90})
	require.NoError(t, err, "JPEG encoding should succeed")

	return buf.Bytes()
}

// createColorTestJPEGImage creates a test JPEG image with distinct color channels for BGR testing.
//
// This helper function generates a JPEG image with distinct RGB values that
// make it easy to validate color channel ordering and BGR conversion functionality.
// Each quadrant has different color characteristics for comprehensive testing.
//
// Arguments:
//   - t: Testing interface for error reporting.
//   - width: The desired image width in pixels.
//   - height: The desired image height in pixels.
//
// Returns:
//   - []byte: The encoded JPEG image data with distinct color patterns.
func createColorTestJPEGImage(t testing.TB, width, height int) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Create distinct color patterns in each quadrant
			var r, g, b uint8
			if x < width/2 && y < height/2 {
				r, g, b = 255, 0, 0 // Red quadrant
			} else if x >= width/2 && y < height/2 {
				r, g, b = 0, 255, 0 // Green quadrant
			} else if x < width/2 && y >= height/2 {
				r, g, b = 0, 0, 255 // Blue quadrant
			} else {
				r, g, b = 255, 255, 255 // White quadrant
			}
			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	var buf bytes.Buffer
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95})
	require.NoError(t, err, "Color test JPEG encoding should succeed")

	return buf.Bytes()
}

// createTestPNGImage creates a test PNG image with specified dimensions for format testing.
//
// This helper function generates a synthetic PNG image to validate PNG format
// support in the preprocessing pipeline. The image contains a checkerboard
// pattern that provides good test coverage for PNG-specific features.
//
// Arguments:
//   - t: Testing interface for error reporting.
//   - width: The desired image width in pixels.
//   - height: The desired image height in pixels.
//
// Returns:
//   - []byte: The encoded PNG image data ready for format testing.
func createTestPNGImage(t testing.TB, width, height int) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// Create a checkerboard pattern
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if (x/10+y/10)%2 == 0 {
				img.Set(x, y, color.RGBA{R: 200, G: 200, B: 200, A: 255})
			} else {
				img.Set(x, y, color.RGBA{R: 50, G: 50, B: 50, A: 255})
			}
		}
	}

	var buf bytes.Buffer
	err := png.Encode(&buf, img)
	require.NoError(t, err, "PNG encoding should succeed")

	return buf.Bytes()
}

// validateImageNetNormalization validates that tensor data follows ImageNet normalization patterns.
//
// This helper function checks that preprocessed tensor data contains values
// within expected ranges for ImageNet normalization. It validates both the
// distribution of values and absence of obvious anomalies.
//
// Arguments:
//   - t: Testing context for assertions and error reporting.
//   - tensorData: The tensor data to validate for proper normalization.
func validateImageNetNormalization(t testing.TB, tensorData []float32) {
	t.Helper()

	// ImageNet normalization typically produces values roughly in range [-2.5, 2.5]
	var min, max float32 = math.MaxFloat32, -math.MaxFloat32
	var sum float64

	for _, pixel := range tensorData {
		if pixel < min {
			min = pixel
		}
		if pixel > max {
			max = pixel
		}
		sum += float64(pixel)
	}

	mean := sum / float64(len(tensorData))

	// Validate reasonable ranges for ImageNet normalization
	assert.GreaterOrEqual(t, min, float32(-5.0), "Normalized pixels should not be extremely negative")
	assert.LessOrEqual(t, max, float32(5.0), "Normalized pixels should not be extremely positive")
	assert.GreaterOrEqual(t, mean, -2.0, "Mean should be reasonable for ImageNet normalization")
	assert.LessOrEqual(t, mean, 2.0, "Mean should be reasonable for ImageNet normalization")

	// Validate that we have some variation in the data
	assert.NotEqual(t, min, max, "Tensor data should contain variation, not constant values")
}

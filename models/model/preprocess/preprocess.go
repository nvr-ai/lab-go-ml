package preprocess

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"math"
	"sync"

	"github.com/pkg/errors"
)

// ImageFormat represents the format of an image.
type ImageFormat string

const (
	// ImageFormatJPEG represents JPEG image format.
	ImageFormatJPEG ImageFormat = "jpeg"
	// ImageFormatPNG represents PNG image format.
	ImageFormatPNG ImageFormat = "png"
)

// Image represents an input image with metadata.
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

// ModelConfig defines preprocessing configuration for a specific model.
type ModelConfig struct {
	// Name of the model for debugging purposes.
	Name string
	// InputWidth is the expected width of the model input.
	InputWidth int
	// InputHeight is the expected height of the model input.
	InputHeight int
	// InputChannels is the number of channels (1 for grayscale, 3 for RGB).
	InputChannels int
	// NormalizationType defines how to normalize pixel values.
	NormalizationType NormalizationType
	// MeanValues for standardization (if NormalizationType is Standardize).
	MeanValues []float32
	// StdValues for standardization (if NormalizationType is Standardize).
	StdValues []float32
	// ChannelOrder defines the channel ordering (CHW or HWC).
	ChannelOrder ChannelOrder
	// ColorMode defines the color space (RGB, BGR, Grayscale).
	ColorMode ColorMode
	// KeepAspectRatio if true, maintains aspect ratio with letterboxing.
	KeepAspectRatio bool
	// LetterboxColor is the color used for letterbox padding (default black).
	LetterboxColor color.Color
	// ApplyDenoise if true, applies denoising to the image.
	ApplyDenoise bool
	// DenoiseStrength controls the strength of denoising (0.0 to 1.0).
	DenoiseStrength float64
}

// NormalizationType defines how pixel values are normalized.
type NormalizationType int

const (
	// NormalizeNone keeps pixel values as 0-255.
	NormalizeNone NormalizationType = iota
	// NormalizeZeroToOne scales pixel values to [0, 1].
	NormalizeZeroToOne
	// NormalizeMinusOneToOne scales pixel values to [-1, 1].
	NormalizeMinusOneToOne
	// NormalizeStandardize applies mean and std normalization.
	NormalizeStandardize
)

// ChannelOrder defines the ordering of image channels.
type ChannelOrder int

const (
	// ChannelOrderCHW is Channel-Height-Width ordering (common for ONNX).
	ChannelOrderCHW ChannelOrder = iota
	// ChannelOrderHWC is Height-Width-Channel ordering.
	ChannelOrderHWC
)

// ColorMode defines the color space of the image.
type ColorMode int

const (
	// ColorModeRGB is standard RGB color mode.
	ColorModeRGB ColorMode = iota
	// ColorModeBGR is BGR color mode (common for OpenCV models).
	ColorModeBGR
	// ColorModeGrayscale is single channel grayscale.
	ColorModeGrayscale
)

// PreprocessingResult contains the preprocessed image data and metadata.
type PreprocessingResult struct {
	// Data is the preprocessed float32 tensor data.
	Data []float32
	// OriginalWidth is the original image width before preprocessing.
	OriginalWidth int
	// OriginalHeight is the original image height before preprocessing.
	OriginalHeight int
	// ScaleX is the horizontal scaling factor applied.
	ScaleX float64
	// ScaleY is the vertical scaling factor applied.
	ScaleY float64
	// PadLeft is the left padding applied for letterboxing.
	PadLeft int
	// PadTop is the top padding applied for letterboxing.
	PadTop int
	// Shape contains the tensor shape [C, H, W] or [H, W, C].
	Shape []int
}

// Preprocessor handles image preprocessing for ONNX models.
type Preprocessor struct {
	config     *ModelConfig
	bufferPool *sync.Pool
	debugMode  bool
}

// NewPreprocessor creates a new preprocessor with the given configuration.
//
// Arguments:
// - config: The model-specific preprocessing configuration.
//
// Returns:
// - A configured Preprocessor instance.
//
// @example
//
//	config := &ModelConfig{
//	    Name:              "yolov4",
//	    InputWidth:        416,
//	    InputHeight:       416,
//	    InputChannels:     3,
//	    NormalizationType: NormalizeZeroToOne,
//	    ChannelOrder:      ChannelOrderCHW,
//	    ColorMode:         ColorModeRGB,
//	    KeepAspectRatio:   true,
//	}
//
// preprocessor := NewPreprocessor(config)
func NewPreprocessor(config *ModelConfig) *Preprocessor {
	// Set default letterbox color if not specified.
	if config.LetterboxColor == nil {
		config.LetterboxColor = color.Black
	}

	return &Preprocessor{
		config: config,
		bufferPool: &sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		},
		debugMode: false,
	}
}

// SetDebugMode enables or disables debug logging.
//
// Arguments:
// - enabled: Whether to enable debug mode.
//
// Returns:
// - None.
//
// @example
// preprocessor.SetDebugMode(true)
func (p *Preprocessor) SetDebugMode(enabled bool) {
	p.debugMode = enabled
}

// Preprocess performs all necessary preprocessing steps on the input image.
//
// Arguments:
// - img: The input image to preprocess.
//
// Returns:
// - PreprocessingResult containing the preprocessed tensor and metadata.
// - error if preprocessing fails.
//
// @example
//
//	img := &Image{
//	    Format: ImageFormatJPEG,
//	    Data:   jpegData,
//	    Width:  1920,
//	    Height: 1080,
//	}
//
// result, err := preprocessor.Preprocess(img)
//
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// tensor := result.Data
func (p *Preprocessor) Preprocess(img *Image) (*PreprocessingResult, error) {
	if p.debugMode {
		fmt.Printf("[DEBUG] Starting preprocessing for model: %s\n", p.config.Name)
		fmt.Printf("[DEBUG] Input image: %dx%d, format: %s\n", img.Width, img.Height, img.Format)
	}

	// Validate input.
	if err := p.validateInput(img); err != nil {
		return nil, errors.Wrap(err, "input validation failed")
	}

	// Decode image.
	decodedImg, err := p.decodeImage(img)
	if err != nil {
		return nil, errors.Wrap(err, "image decoding failed")
	}

	// Store original dimensions.
	originalWidth := decodedImg.Bounds().Dx()
	originalHeight := decodedImg.Bounds().Dy()

	if p.debugMode {
		fmt.Printf("[DEBUG] Decoded image dimensions: %dx%d\n", originalWidth, originalHeight)
	}

	// Apply denoising if configured.
	if p.config.ApplyDenoise {
		decodedImg = p.denoise(decodedImg)
		if p.debugMode {
			fmt.Printf("[DEBUG] Applied denoising with strength: %.2f\n", p.config.DenoiseStrength)
		}
	}

	// Convert color mode if necessary.
	decodedImg = p.convertColorMode(decodedImg)

	// Resize image.
	resizedImg, scaleX, scaleY, padLeft, padTop := p.resizeImage(decodedImg)

	if p.debugMode {
		fmt.Printf("[DEBUG] Resized to: %dx%d, scale: (%.4f, %.4f), padding: (%d, %d)\n",
			p.config.InputWidth, p.config.InputHeight, scaleX, scaleY, padLeft, padTop)
	}

	// Convert to tensor.
	tensor := p.imageToTensor(resizedImg)

	// Apply normalization.
	p.normalize(tensor)

	// Determine shape based on channel ordering.
	var shape []int
	if p.config.ChannelOrder == ChannelOrderCHW {
		shape = []int{p.config.InputChannels, p.config.InputHeight, p.config.InputWidth}
	} else {
		shape = []int{p.config.InputHeight, p.config.InputWidth, p.config.InputChannels}
	}

	if p.debugMode {
		fmt.Printf("[DEBUG] Output tensor shape: %v\n", shape)
		fmt.Printf("[DEBUG] Preprocessing complete\n")
	}

	return &PreprocessingResult{
		Data:           tensor,
		OriginalWidth:  originalWidth,
		OriginalHeight: originalHeight,
		ScaleX:         scaleX,
		ScaleY:         scaleY,
		PadLeft:        padLeft,
		PadTop:         padTop,
		Shape:          shape,
	}, nil
}

// validateInput validates the input image structure.
//
// Arguments:
// - img: The image to validate.
//
// Returns:
// - error if validation fails.
//
// @example
// err := preprocessor.validateInput(img)
//
//	if err != nil {
//	    return err
//	}
func (p *Preprocessor) validateInput(img *Image) error {
	if img == nil {
		return errors.New("image is nil")
	}
	if len(img.Data) == 0 {
		return errors.New("image data is empty")
	}
	if img.Width <= 0 || img.Height <= 0 {
		return fmt.Errorf("invalid image dimensions: %dx%d", img.Width, img.Height)
	}
	return nil
}

// decodeImage decodes the image data into an image.Image.
//
// Arguments:
// - img: The image to decode.
//
// Returns:
// - The decoded image.
// - error if decoding fails.
//
// @example
// decodedImg, err := preprocessor.decodeImage(img)
//
//	if err != nil {
//	    return nil, err
//	}
func (p *Preprocessor) decodeImage(img *Image) (image.Image, error) {
	buf := p.bufferPool.Get().(*bytes.Buffer)
	defer func() {
		buf.Reset()
		p.bufferPool.Put(buf)
	}()

	buf.Write(img.Data)
	reader := bytes.NewReader(buf.Bytes())

	// Decode based on format.
	switch img.Format {
	case ImageFormatJPEG:
		return jpeg.Decode(reader)
	default:
		// Try auto-detection.
		reader.Seek(0, 0)
		decoded, _, err := image.Decode(reader)
		return decoded, err
	}
}

// denoise applies denoising to the image.
//
// Arguments:
// - img: The image to denoise.
//
// Returns:
// - The denoised image.
//
// @example
// denoisedImg := preprocessor.denoise(inputImg)
func (p *Preprocessor) denoise(img image.Image) image.Image {
	// Apply Gaussian blur for denoising.
	sigma := p.config.DenoiseStrength * 2.0
	if sigma <= 0 {
		sigma = 0.5
	}
	return images.Blur(img, sigma)
}

// convertColorMode converts the image to the required color mode.
//
// Arguments:
// - img: The image to convert.
//
// Returns:
// - The converted image.
//
// @example
// convertedImg := preprocessor.convertColorMode(inputImg)
func (p *Preprocessor) convertColorMode(img image.Image) image.Image {
	switch p.config.ColorMode {
	case ColorModeGrayscale:
		return images.Grayscale(img)
	case ColorModeBGR:
		// For BGR, we'll handle this in the tensor conversion.
		return img
	default:
		return img
	}
}

// resizeImage resizes the image to the model's input dimensions.
//
// Arguments:
// - img: The image to resize.
//
// Returns:
// - The resized image.
// - scaleX: Horizontal scaling factor.
// - scaleY: Vertical scaling factor.
// - padLeft: Left padding for letterboxing.
// - padTop: Top padding for letterboxing.
//
// @example
// resized, scaleX, scaleY, padLeft, padTop := preprocessor.resizeImage(img)
func (p *Preprocessor) resizeImage(img image.Image) (image.Image, float64, float64, int, int) {
	bounds := img.Bounds()
	srcWidth := bounds.Dx()
	srcHeight := bounds.Dy()

	if !p.config.KeepAspectRatio {
		// Simple resize without maintaining aspect ratio.
		resized := images.Resize(img, p.config.InputWidth, p.config.InputHeight, images.Lanczos)
		scaleX := float64(p.config.InputWidth) / float64(srcWidth)
		scaleY := float64(p.config.InputHeight) / float64(srcHeight)
		return resized, scaleX, scaleY, 0, 0
	}

	// Calculate scale to maintain aspect ratio.
	scaleX := float64(p.config.InputWidth) / float64(srcWidth)
	scaleY := float64(p.config.InputHeight) / float64(srcHeight)
	scale := math.Min(scaleX, scaleY)

	// Calculate new dimensions.
	newWidth := int(float64(srcWidth) * scale)
	newHeight := int(float64(srcHeight) * scale)

	// Resize image.
	resized := images.Resize(img, newWidth, newHeight, images.Lanczos)

	// Calculate padding.
	padLeft := (p.config.InputWidth - newWidth) / 2
	padTop := (p.config.InputHeight - newHeight) / 2

	// Create letterboxed image.
	letterboxed := image.NewRGBA(image.Rect(0, 0, p.config.InputWidth, p.config.InputHeight))
	draw.Draw(letterboxed, letterboxed.Bounds(), &image.Uniform{p.config.LetterboxColor}, image.Point{}, draw.Src)
	draw.Draw(letterboxed, image.Rect(padLeft, padTop, padLeft+newWidth, padTop+newHeight),
		resized, image.Point{}, draw.Over)

	return letterboxed, scale, scale, padLeft, padTop
}

// imageToTensor converts an image to a float32 tensor.
//
// Arguments:
// - img: The image to convert.
//
// Returns:
// - The float32 tensor data.
//
// @example
// tensor := preprocessor.imageToTensor(img)
func (p *Preprocessor) imageToTensor(img image.Image) []float32 {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Allocate tensor.
	tensorSize := width * height * p.config.InputChannels
	tensor := make([]float32, tensorSize)

	// Convert based on channel ordering and color mode.
	idx := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			// Convert from uint32 to uint8.
			r8 := uint8(r >> 8)
			g8 := uint8(g >> 8)
			b8 := uint8(b >> 8)

			if p.config.InputChannels == 1 {
				// Grayscale.
				gray := 0.299*float32(r8) + 0.587*float32(g8) + 0.114*float32(b8)
				if p.config.ChannelOrder == ChannelOrderCHW {
					tensor[y*width+x] = gray
				} else {
					tensor[idx] = gray
					idx++
				}
			} else {
				// RGB or BGR.
				var ch0, ch1, ch2 float32
				if p.config.ColorMode == ColorModeBGR {
					ch0, ch1, ch2 = float32(b8), float32(g8), float32(r8)
				} else {
					ch0, ch1, ch2 = float32(r8), float32(g8), float32(b8)
				}

				if p.config.ChannelOrder == ChannelOrderCHW {
					// CHW format.
					tensor[0*height*width+y*width+x] = ch0
					tensor[1*height*width+y*width+x] = ch1
					tensor[2*height*width+y*width+x] = ch2
				} else {
					// HWC format.
					tensor[idx] = ch0
					tensor[idx+1] = ch1
					tensor[idx+2] = ch2
					idx += 3
				}
			}
		}
	}

	return tensor
}

// normalize applies normalization to the tensor.
//
// Arguments:
// - tensor: The tensor to normalize in-place.
//
// Returns:
// - None (modifies tensor in-place).
//
// @example
// preprocessor.normalize(tensor)
func (p *Preprocessor) normalize(tensor []float32) {
	switch p.config.NormalizationType {
	case NormalizeZeroToOne:
		for i := range tensor {
			tensor[i] /= 255.0
		}
	case NormalizeMinusOneToOne:
		for i := range tensor {
			tensor[i] = (tensor[i] / 127.5) - 1.0
		}
	case NormalizeStandardize:
		if len(p.config.MeanValues) != p.config.InputChannels ||
			len(p.config.StdValues) != p.config.InputChannels {
			// Fallback to zero-to-one if mean/std not properly configured.
			for i := range tensor {
				tensor[i] /= 255.0
			}
			return
		}

		// Apply channel-wise standardization.
		pixelsPerChannel := len(tensor) / p.config.InputChannels
		for c := 0; c < p.config.InputChannels; c++ {
			mean := p.config.MeanValues[c]
			std := p.config.StdValues[c]

			if p.config.ChannelOrder == ChannelOrderCHW {
				// CHW format.
				offset := c * pixelsPerChannel
				for i := 0; i < pixelsPerChannel; i++ {
					tensor[offset+i] = (tensor[offset+i] - mean) / std
				}
			} else {
				// HWC format.
				for i := c; i < len(tensor); i += p.config.InputChannels {
					tensor[i] = (tensor[i] - mean) / std
				}
			}
		}
	}
}

// GetYOLOv4Config returns a standard configuration for YOLOv4 models.
//
// Arguments:
// - inputSize: The input size (typically 416, 512, or 608).
//
// Returns:
// - A configured ModelConfig for YOLOv4.
//
// @example
// config := GetYOLOv4Config(416)
// preprocessor := NewPreprocessor(config)
func GetYOLOv4Config(inputSize int) *ModelConfig {
	return &ModelConfig{
		Name:              "yolov4",
		InputWidth:        inputSize,
		InputHeight:       inputSize,
		InputChannels:     3,
		NormalizationType: NormalizeZeroToOne,
		ChannelOrder:      ChannelOrderCHW,
		ColorMode:         ColorModeRGB,
		KeepAspectRatio:   true,
		LetterboxColor:    color.RGBA{114, 114, 114, 255},
		ApplyDenoise:      false,
	}
}

// GetDFineConfig returns a standard configuration for D-FINE models.
//
// Arguments:
// - inputSize: The input size for the model.
//
// Returns:
// - A configured ModelConfig for D-FINE.
//
// @example
// config := GetDFineConfig(640)
// preprocessor := NewPreprocessor(config)
func GetDFineConfig(inputSize int) *ModelConfig {
	return &ModelConfig{
		Name:              "d-fine",
		InputWidth:        inputSize,
		InputHeight:       inputSize,
		InputChannels:     3,
		NormalizationType: NormalizeStandardize,
		MeanValues:        []float32{123.675, 116.28, 103.53},
		StdValues:         []float32{58.395, 57.12, 57.375},
		ChannelOrder:      ChannelOrderCHW,
		ColorMode:         ColorModeRGB,
		KeepAspectRatio:   true,
		LetterboxColor:    color.Black,
		ApplyDenoise:      false,
	}
}

// GetRTDETRConfig returns a standard configuration for RT-DETR models.
//
// Arguments:
// - inputSize: The input size for the model.
//
// Returns:
// - A configured ModelConfig for RT-DETR.
//
// @example
// config := GetRTDETRConfig(640)
// preprocessor := NewPreprocessor(config)
func GetRTDETRConfig(inputSize int) *ModelConfig {
	return &ModelConfig{
		Name:              "rt-detr",
		InputWidth:        inputSize,
		InputHeight:       inputSize,
		InputChannels:     3,
		NormalizationType: NormalizeStandardize,
		MeanValues:        []float32{103.53, 116.28, 123.675},
		StdValues:         []float32{57.375, 57.12, 58.395},
		ChannelOrder:      ChannelOrderCHW,
		ColorMode:         ColorModeBGR,
		KeepAspectRatio:   true,
		LetterboxColor:    color.Black,
		ApplyDenoise:      false,
	}
}

// BatchPreprocess processes multiple images in parallel.
//
// Arguments:
// - images: Slice of images to preprocess.
// - maxConcurrency: Maximum number of images to process concurrently.
//
// Returns:
// - Slice of preprocessing results.
// - error if any preprocessing fails.
//
// @example
// images := []*Image{img1, img2, img3}
// results, err := preprocessor.BatchPreprocess(images, 4)
//
//	if err != nil {
//	    log.Fatal(err)
//	}
func (p *Preprocessor) BatchPreprocess(images []*Image, maxConcurrency int) ([]*PreprocessingResult, error) {
	if maxConcurrency <= 0 {
		maxConcurrency = 1
	}

	results := make([]*PreprocessingResult, len(images))
	errors := make([]error, len(images))

	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for i, img := range images {
		wg.Add(1)
		go func(idx int, image *Image) {
			defer wg.Done()

			sem <- struct{}{}
			defer func() { <-sem }()

			result, err := p.Preprocess(image)
			if err != nil {
				errors[idx] = fmt.Errorf("failed to preprocess image %d: %w", idx, err)
			} else {
				results[idx] = result
			}
		}(i, img)
	}

	wg.Wait()

	// Check for errors.
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

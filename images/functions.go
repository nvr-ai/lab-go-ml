// Package images - provides high-performance, idempotent image processing operations
// optimized for machine learning preprocessing pipelines.
package images

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"runtime"
	"sync"
)

// ResampleFilter defines the resampling algorithm used for image scaling.
type ResampleFilter int

const (
	// NearestNeighborFilter uses nearest-neighbor interpolation (fastest, lowest quality).
	NearestNeighborFilter ResampleFilter = iota
	// BilinearFilter uses bilinear interpolation (fast, good quality).
	BilinearFilter
	// BicubicFilter uses bicubic interpolation (slower, better quality).
	BicubicFilter
	// LanczosFilter uses LanczosFilter resampling with a=3 (slowest, best quality).
	LanczosFilter
	// MitchellNetravaliFilter uses Mitchell-Netravali cubic filter (balanced).
	MitchellNetravaliFilter
)

// kernel represents a resampling kernel function.
type kernel struct {
	// Support is the support radius which is the radius of the kernel in pixels.
	Support float64
	// At evaluates the kernel at distance x. This is the function that is used to
	// calculate the weight of the pixel at the given distance.
	At func(x float64) float64
}

// kernels maps each filter type to its kernel function.
var kernels = map[ResampleFilter]kernel{
	NearestNeighborFilter: {
		Support: 0.5,
		At: func(x float64) float64 {
			// Nearest neighbor returns 1.0 for distances within 0.5, 0.0 otherwise.
			// This ensures proper sampling of the closest pixel.
			if math.Abs(x) < 0.5 {
				return 1.0
			}
			return 0.0
		},
	},
	BilinearFilter: {
		Support: 1.0,
		At: func(x float64) float64 {
			// Bilinear kernel is a triangle function.
			// Returns linear interpolation weight based on distance.
			x = math.Abs(x)
			if x < 1.0 {
				return 1.0 - x
			}
			return 0.0
		},
	},
	BicubicFilter: {
		Support: 2.0,
		At: func(x float64) float64 {
			// Mitchell-Netravali cubic with B=0, C=0.5 (Catmull-Rom).
			// Provides smooth interpolation with minimal ringing.
			x = math.Abs(x)
			if x < 1.0 {
				return (1.5*x-2.5)*x*x + 1.0
			}
			if x < 2.0 {
				return ((-0.5*x+2.5)*x-4.0)*x + 2.0
			}
			return 0.0
		},
	},
	LanczosFilter: {
		Support: 3.0,
		At: func(x float64) float64 {
			// Lanczos kernel with a=3.
			// Provides excellent sharpness but can introduce ringing artifacts.
			if x == 0.0 {
				return 1.0
			}
			x = math.Abs(x)
			if x >= 3.0 {
				return 0.0
			}
			// sinc(x) * sinc(x/3)
			pix := math.Pi * x
			return (math.Sin(pix) / pix) * (math.Sin(pix/3.0) / (pix / 3.0))
		},
	},
	MitchellNetravaliFilter: {
		Support: 2.0,
		At: func(x float64) float64 {
			// Mitchell-Netravali with B=1/3, C=1/3.
			// Balanced between sharpness and ringing suppression.
			x = math.Abs(x)
			if x < 1.0 {
				return ((1.16666666666667*x-2.0)*x)*x + 0.888888888888889
			}
			if x < 2.0 {
				return ((-0.388888888888889*x+2.0)*x-3.333333333333333)*x + 1.777777777777778
			}
			return 0.0
		},
	},
}

// Contribution represents a single pixel's contribution to the output.
type Contribution struct {
	// pixel is the source pixel index.
	pixel int
	// weight is the contribution weight.
	weight float64
}

// NearestNeighbor returns the weight for Nearest Neighbor interpolation.
// This kernel is used when speed is prioritized over quality.
// It returns 1.0 if the distance is zero (exact pixel match), otherwise 0.0.
//
// Even though Nearest Neighbor is rarely used in preprocessing due to its poor
// quality, it's still important to implement it correctly—especially for
// debugging, visual inspection, or edge deployments where performance trumps
// fidelity.
//
// Arguments:
// - x: The distance from the center pixel.
//
// Returns:
// - The weight for the pixel at the given distance.
//
// Example:
// ```go`
// weight := NearestNeighbor(0.5) // Returns 1.0
// ````
func NearestNeighbor(x float64) float64 {
	if x == 0 {
		return 1.0 // Exact match: full contribution
	}
	return 0.0 // All other pixels: no contribution
}

// Resize performs high-quality image resizing using the specified resampling filter.
// This implementation uses separable filtering for efficiency, processing
// horizontal and vertical dimensions independently.
//
// Arguments:
// - img: The source image to resize.
// - width: The target width in pixels.
// - height: The target height in pixels.
// - filter: The resampling filter to use for interpolation.
//
// Returns:
// - The resized image maintaining the original color model.
//
// @example
// resized := Resize(srcImage, 224, 224, Lanczos)
// resized := Resize(srcImage, 416, 416, Bilinear)
func Resize(img image.Image, width, height int, filter ResampleFilter) image.Image {
	// Early return for invalid dimensions.
	if width <= 0 || height <= 0 {
		// Return a minimal 1x1 image to maintain idempotency.
		return image.NewRGBA(image.Rect(0, 0, 1, 1))
	}

	// Get source image bounds.
	bounds := img.Bounds()
	srcWidth := bounds.Dx()
	srcHeight := bounds.Dy()

	// Early return if no resizing needed (idempotency optimization).
	if srcWidth == width && srcHeight == height {
		// Create a copy to ensure idempotency (caller gets a new image).
		dst := image.NewRGBA(image.Rect(0, 0, width, height))
		draw.Draw(dst, dst.Bounds(), img, bounds.Min, draw.Src)
		return dst
	}

	// Handle nearest neighbor separately for maximum performance.
	if filter == NearestNeighborFilter {
		return ResizeNearestNeighbor(img, width, height)
	}

	// Create intermediate image for horizontal pass.
	// We resize horizontally first, then vertically (separable filtering).
	intermediate := image.NewRGBA(image.Rect(0, 0, width, srcHeight))

	// Perform horizontal resize.
	ResizeHorizontal(img, intermediate, filter)

	// Create final output image.
	dst := image.NewRGBA(image.Rect(0, 0, width, height))

	// Perform vertical resize.
	ResizeVertical(intermediate, dst, filter)

	return dst
}

// ResizeNearestNeighbor performs fast nearest-neighbor resizing.
// This is the fastest resampling method but produces blocky results.
//
// Arguments:
// - src: The source image.
// - width: Target width.
// - height: Target height.
//
// Returns:
// - The resized image using nearest-neighbor sampling.
//
// @example
// fast := ResizeNearestNeighbor(src, 224, 224)
func ResizeNearestNeighbor(src image.Image, width, height int) image.Image {
	// Get source bounds.
	bounds := src.Bounds()
	srcWidth := bounds.Dx()
	srcHeight := bounds.Dy()

	// Create destination image.
	dst := image.NewRGBA(image.Rect(0, 0, width, height))

	// Calculate scaling ratios.
	// We use floating point to maintain precision during calculation.
	xRatio := float64(srcWidth) / float64(width)
	yRatio := float64(srcHeight) / float64(height)

	// Process pixels in parallel for better performance.
	Parallel(height, func(partStart, partEnd int) {
		// Process each row in this partition.
		for y := partStart; y < partEnd; y++ {
			// Calculate source Y coordinate.
			srcY := int(float64(y)*yRatio + 0.5)
			// Clamp to valid range.
			if srcY >= srcHeight {
				srcY = srcHeight - 1
			}

			// Process each pixel in the row.
			for x := 0; x < width; x++ {
				// Calculate source X coordinate.
				srcX := int(float64(x)*xRatio + 0.5)
				// Clamp to valid range.
				if srcX >= srcWidth {
					srcX = srcWidth - 1
				}

				// Copy pixel from source to destination.
				dst.Set(x, y, src.At(bounds.Min.X+srcX, bounds.Min.Y+srcY))
			}
		}
	})

	return dst
}

// ResizeHorizontal performs horizontal resizing using the specified filter.
// This is the first pass of separable filtering.
//
// Arguments:
// - src: Source image.
// - dst: Destination image (must have correct width, same height as source).
// - filter: Resampling filter to use.
//
// Returns:
// - None (modifies dst in-place).
//
// @example
// intermediate := image.NewRGBA(image.Rect(0, 0, newWidth, srcHeight))
// ResizeHorizontal(src, intermediate, Lanczos)
func ResizeHorizontal(src image.Image, dst *image.RGBA, filter ResampleFilter) {
	// Get image dimensions.
	srcBounds := src.Bounds()
	dstBounds := dst.Bounds()
	srcWidth := srcBounds.Dx()
	dstWidth := dstBounds.Dx()
	height := srcBounds.Dy()

	// Get kernel for the specified filter.
	k := kernels[filter]

	// Calculate scaling ratio.
	scale := float64(srcWidth) / float64(dstWidth)

	// Calculate filter scale and support.
	// When downsampling, we need to expand the filter support.
	filterScale := math.Max(scale, 1.0)
	support := k.Support * filterScale

	// Pre-calculate contributions for each output column.
	// This avoids redundant calculations in the inner loop.
	contributions := make([][]Contribution, dstWidth)

	// Calculate contributions for each destination pixel.
	for x := 0; x < dstWidth; x++ {
		// Calculate center position in source image.
		center := (float64(x) + 0.5) * scale

		// Calculate contributing pixel range.
		left := int(math.Floor(center - support))
		right := int(math.Ceil(center + support))

		// Clamp to valid range.
		if left < 0 {
			left = 0
		}
		if right >= srcWidth {
			right = srcWidth - 1
		}

		// Calculate weights for contributing pixels.
		var weights []Contribution
		var sum float64

		for srcX := left; srcX <= right; srcX++ {
			// Calculate normalized distance from center.
			distance := math.Abs(float64(srcX) - center + 0.5)

			// Calculate weight using kernel function.
			weight := k.At(distance / filterScale)

			if weight != 0 {
				weights = append(weights, Contribution{
					pixel:  srcX,
					weight: weight,
				})
				sum += weight
			}
		}

		// Normalize weights to sum to 1.0.
		// This ensures brightness preservation.
		if sum != 0 {
			for i := range weights {
				weights[i].weight /= sum
			}
		}

		contributions[x] = weights
	}

	// Process rows in parallel for better performance.
	Parallel(height, func(partStart, partEnd int) {
		// Process each row in this partition.
		for y := partStart; y < partEnd; y++ {
			srcY := srcBounds.Min.Y + y

			// Process each output pixel in the row.
			for x := 0; x < dstWidth; x++ {
				// Initialize accumulator for RGBA channels.
				var r, g, b, a float64

				// Sum contributions from source pixels.
				for _, c := range contributions[x] {
					srcX := srcBounds.Min.X + c.pixel

					// Get source pixel color.
					srcColor := src.At(srcX, srcY)
					srcR, srcG, srcB, srcA := srcColor.RGBA()

					// Accumulate weighted contribution.
					// Note: RGBA() returns 16-bit values, we normalize to 8-bit.
					weight := c.weight
					r += float64(srcR>>8) * weight
					g += float64(srcG>>8) * weight
					b += float64(srcB>>8) * weight
					a += float64(srcA>>8) * weight
				}

				// Clamp values to valid range [0, 255].
				r = Clamp(r, 0, 255)
				g = Clamp(g, 0, 255)
				b = Clamp(b, 0, 255)
				a = Clamp(a, 0, 255)

				// Set destination pixel.
				dst.SetRGBA(x, y, color.RGBA{
					R: uint8(r + 0.5), // Round to nearest integer.
					G: uint8(g + 0.5),
					B: uint8(b + 0.5),
					A: uint8(a + 0.5),
				})
			}
		}
	})
}

// ResizeVertical performs vertical resizing using the specified filter.
// This is the second pass of separable filtering.
//
// Arguments:
// - src: Source image (typically the output of horizontal resize).
// - dst: Destination image (must have correct dimensions).
// - filter: Resampling filter to use.
//
// Returns:
// - None (modifies dst in-place).
//
// @example
// dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
// ResizeVertical(intermediate, dst, Lanczos)
func ResizeVertical(src *image.RGBA, dst *image.RGBA, filter ResampleFilter) {
	// Get image dimensions.
	srcBounds := src.Bounds()
	dstBounds := dst.Bounds()
	srcHeight := srcBounds.Dy()
	dstHeight := dstBounds.Dy()
	width := dstBounds.Dx()

	// Get kernel for the specified filter.
	k := kernels[filter]

	// Calculate scaling ratio.
	scale := float64(srcHeight) / float64(dstHeight)

	// Calculate filter scale and support.
	filterScale := math.Max(scale, 1.0)
	support := k.Support * filterScale

	// Pre-calculate contributions for each output row.
	contributions := make([][]Contribution, dstHeight)

	// Calculate contributions for each destination pixel.
	for y := 0; y < dstHeight; y++ {
		// Calculate center position in source image.
		center := (float64(y) + 0.5) * scale

		// Calculate contributing pixel range.
		top := int(math.Floor(center - support))
		bottom := int(math.Ceil(center + support))

		// Clamp to valid range.
		if top < 0 {
			top = 0
		}
		if bottom >= srcHeight {
			bottom = srcHeight - 1
		}

		// Calculate weights for contributing pixels.
		var weights []Contribution
		var sum float64

		for srcY := top; srcY <= bottom; srcY++ {
			// Calculate normalized distance from center.
			distance := math.Abs(float64(srcY) - center + 0.5)

			// Calculate weight using kernel function.
			weight := k.At(distance / filterScale)

			if weight != 0 {
				weights = append(weights, Contribution{
					pixel:  srcY,
					weight: weight,
				})
				sum += weight
			}
		}

		// Normalize weights.
		if sum != 0 {
			for i := range weights {
				weights[i].weight /= sum
			}
		}

		contributions[y] = weights
	}

	// Process columns in parallel for better performance.
	Parallel(width, func(partStart, partEnd int) {
		// Process each column in this partition.
		for x := partStart; x < partEnd; x++ {
			// Process each output pixel in the column.
			for y := 0; y < dstHeight; y++ {
				// Initialize accumulator for RGBA channels.
				var r, g, b, a float64

				// Sum contributions from source pixels.
				for _, c := range contributions[y] {
					srcY := c.pixel

					// Get source pixel directly from RGBA image (fast path).
					srcIdx := src.PixOffset(x, srcY)

					// Accumulate weighted contribution.
					weight := c.weight
					r += float64(src.Pix[srcIdx+0]) * weight
					g += float64(src.Pix[srcIdx+1]) * weight
					b += float64(src.Pix[srcIdx+2]) * weight
					a += float64(src.Pix[srcIdx+3]) * weight
				}

				// Clamp values to valid range.
				r = Clamp(r, 0, 255)
				g = Clamp(g, 0, 255)
				b = Clamp(b, 0, 255)
				a = Clamp(a, 0, 255)

				// Set destination pixel directly (fast path).
				dstIdx := dst.PixOffset(x, y)
				dst.Pix[dstIdx+0] = uint8(r + 0.5)
				dst.Pix[dstIdx+1] = uint8(g + 0.5)
				dst.Pix[dstIdx+2] = uint8(b + 0.5)
				dst.Pix[dstIdx+3] = uint8(a + 0.5)
			}
		}
	})
}

// Grayscale converts an image to grayscale using ITU-R BT.709 luma coefficients.
// This provides perceptually accurate grayscale conversion.
//
// Arguments:
// - img: The source image to convert.
//
// Returns:
// - A new grayscale image with the same dimensions.
//
// @example
// gray := Grayscale(colorImage)
func Grayscale(img image.Image) image.Image {
	// Get image bounds.
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Create destination grayscale image.
	// We use RGBA for compatibility, but all channels will have the same value.
	dst := image.NewRGBA(bounds)

	// ITU-R BT.709 luma coefficients.
	// These weights reflect human eye sensitivity to different colors.
	const (
		redWeight   = 0.2126 // Red contributes ~21% to perceived brightness.
		greenWeight = 0.7152 // Green contributes ~72% to perceived brightness.
		blueWeight  = 0.0722 // Blue contributes ~7% to perceived brightness.
	)

	// Process pixels in parallel for better performance.
	Parallel(height, func(partStart, partEnd int) {
		// Process each row in this partition.
		for y := partStart; y < partEnd; y++ {
			srcY := bounds.Min.Y + y

			// Process each pixel in the row.
			for x := 0; x < width; x++ {
				srcX := bounds.Min.X + x

				// Get source pixel color.
				c := img.At(srcX, srcY)
				r, g, b, a := c.RGBA()

				// Calculate luminance using BT.709 coefficients.
				// Note: RGBA() returns 16-bit values, we work in that space for precision.
				luma := uint32(float64(r)*redWeight + float64(g)*greenWeight + float64(b)*blueWeight)

				// Convert back to 8-bit for storage.
				gray := uint8(luma >> 8)

				// Set all RGB channels to the same gray value.
				// Alpha channel is preserved from the source.
				dst.SetRGBA(x, y, color.RGBA{
					R: gray,
					G: gray,
					B: gray,
					A: uint8(a >> 8),
				})
			}
		}
	})

	return dst
}

// Blur applies Gaussian blur to an image for noise reduction or smoothing.
// This implementation uses separable filtering for efficiency.
//
// Arguments:
// - img: The source image to blur.
// - sigma: Standard deviation of the Gaussian kernel (controls blur strength).
//
// Returns:
// - A new blurred image with the same dimensions.
//
// @example
// blurred := Blur(noisyImage, 1.5)
// denoised := Blur(img, 0.5)
func Blur(img image.Image, sigma float64) image.Image {
	// Clamp sigma to a reasonable range.
	// Too small values have no effect, too large values are computationally expensive.
	if sigma <= 0 {
		sigma = 0.1
	}
	if sigma > 10.0 {
		sigma = 10.0
	}

	// For very small sigma, return a copy of the original (optimization).
	if sigma < 0.1 {
		bounds := img.Bounds()
		dst := image.NewRGBA(bounds)
		draw.Draw(dst, bounds, img, bounds.Min, draw.Src)
		return dst
	}

	// Calculate kernel radius based on sigma.
	// We use 3*sigma to capture 99.7% of the Gaussian distribution.
	radius := int(math.Ceil(sigma * 3.0))

	// Generate 1D Gaussian kernel.
	kernel := GenerateGaussianKernel(radius, sigma)

	// Get image bounds.
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Create intermediate image for horizontal pass.
	intermediate := image.NewRGBA(bounds)

	// Apply horizontal blur.
	BlurHorizontal(img, intermediate, kernel)

	// Create final output image.
	dst := image.NewRGBA(bounds)

	// Apply vertical blur.
	BlurVertical(intermediate, dst, kernel)

	return dst
}

// GenerateGaussianKernel creates a 1D Gaussian kernel for separable filtering.
// The kernel is normalized to sum to 1.0.
//
// Arguments:
// - radius: The kernel radius (kernel size will be 2*radius + 1).
// - sigma: Standard deviation of the Gaussian.
//
// Returns:
// - A normalized 1D Gaussian kernel.
//
// @example
// kernel := GenerateGaussianKernel(3, 1.5)
func GenerateGaussianKernel(radius int, sigma float64) []float64 {
	// Kernel size is 2*radius + 1 (includes center pixel).
	size := 2*radius + 1
	kernel := make([]float64, size)

	// Pre-calculate constant factor.
	// This is 1/(sqrt(2*pi)*sigma), the normalization factor for Gaussian.
	factor := 1.0 / (math.Sqrt(2.0*math.Pi) * sigma)

	// Pre-calculate denominator for exponent.
	// This is 2*sigma^2, used in the Gaussian formula.
	denom := 2.0 * sigma * sigma

	// Calculate kernel values.
	sum := 0.0
	for i := 0; i < size; i++ {
		// Calculate distance from center.
		x := float64(i - radius)

		// Calculate Gaussian value: exp(-(x^2)/(2*sigma^2)).
		kernel[i] = factor * math.Exp(-(x*x)/denom)

		// Accumulate sum for normalization.
		sum += kernel[i]
	}

	// Normalize kernel to sum to 1.0.
	// This ensures the blur doesn't change image brightness.
	for i := range kernel {
		kernel[i] /= sum
	}

	return kernel
}

// BlurHorizontal applies horizontal Gaussian blur using the provided kernel.
// This is the first pass of separable Gaussian filtering.
//
// Arguments:
// - src: Source image.
// - dst: Destination image (must have same dimensions as source).
// - kernel: 1D Gaussian kernel.
//
// Returns:
// - None (modifies dst in-place).
//
// @example
// intermediate := image.NewRGBA(bounds)
// BlurHorizontal(src, intermediate, kernel)
func BlurHorizontal(src image.Image, dst *image.RGBA, kernel []float64) {
	// Get image dimensions.
	bounds := src.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Calculate kernel radius.
	radius := len(kernel) / 2

	// Process rows in parallel for better performance.
	Parallel(height, func(partStart, partEnd int) {
		// Process each row in this partition.
		for y := partStart; y < partEnd; y++ {
			srcY := bounds.Min.Y + y

			// Process each pixel in the row.
			for x := 0; x < width; x++ {
				// Initialize accumulators for RGBA channels.
				var r, g, b, a float64

				// Apply kernel to surrounding pixels.
				for i, weight := range kernel {
					// Calculate source X coordinate.
					srcX := x + i - radius

					// Handle border pixels using clamp-to-edge strategy.
					// This prevents dark borders and maintains edge information.
					if srcX < 0 {
						srcX = 0
					} else if srcX >= width {
						srcX = width - 1
					}

					// Get source pixel color.
					c := src.At(bounds.Min.X+srcX, srcY)
					srcR, srcG, srcB, srcA := c.RGBA()

					// Accumulate weighted contribution.
					r += float64(srcR>>8) * weight
					g += float64(srcG>>8) * weight
					b += float64(srcB>>8) * weight
					a += float64(srcA>>8) * weight
				}

				// Set destination pixel with blurred values.
				dst.SetRGBA(x, y, color.RGBA{
					R: uint8(Clamp(r, 0, 255) + 0.5),
					G: uint8(Clamp(g, 0, 255) + 0.5),
					B: uint8(Clamp(b, 0, 255) + 0.5),
					A: uint8(Clamp(a, 0, 255) + 0.5),
				})
			}
		}
	})
}

// BlurVertical applies vertical Gaussian blur using the provided kernel.
// This is the second pass of separable Gaussian filtering.
//
// Arguments:
// - src: Source image (typically the output of horizontal blur).
// - dst: Destination image (must have same dimensions as source).
// - kernel: 1D Gaussian kernel.
//
// Returns:
// - None (modifies dst in-place).
//
// @example
// dst := image.NewRGBA(bounds)
// BlurVertical(intermediate, dst, kernel)
func BlurVertical(src *image.RGBA, dst *image.RGBA, kernel []float64) {
	// Get image dimensions.
	bounds := src.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Calculate kernel radius.
	radius := len(kernel) / 2

	// Process columns in parallel for better performance.
	Parallel(width, func(partStart, partEnd int) {
		// Process each column in this partition.
		for x := partStart; x < partEnd; x++ {
			// Process each pixel in the column.
			for y := 0; y < height; y++ {
				// Initialize accumulators for RGBA channels.
				var r, g, b, a float64

				// Apply kernel to surrounding pixels.
				for i, weight := range kernel {
					// Calculate source Y coordinate.
					srcY := y + i - radius

					// Handle border pixels using clamp-to-edge strategy.
					if srcY < 0 {
						srcY = 0
					} else if srcY >= height {
						srcY = height - 1
					}

					// Get source pixel directly from RGBA image (fast path).
					// This avoids the overhead of the At() interface.
					srcIdx := src.PixOffset(x, srcY)

					// Accumulate weighted contribution.
					r += float64(src.Pix[srcIdx+0]) * weight
					g += float64(src.Pix[srcIdx+1]) * weight
					b += float64(src.Pix[srcIdx+2]) * weight
					a += float64(src.Pix[srcIdx+3]) * weight
				}

				// Set destination pixel directly (fast path).
				dstIdx := dst.PixOffset(x, y)
				dst.Pix[dstIdx+0] = uint8(Clamp(r, 0, 255) + 0.5)
				dst.Pix[dstIdx+1] = uint8(Clamp(g, 0, 255) + 0.5)
				dst.Pix[dstIdx+2] = uint8(Clamp(b, 0, 255) + 0.5)
				dst.Pix[dstIdx+3] = uint8(Clamp(a, 0, 255) + 0.5)
			}
		}
	})
}

// Clamp restricts a value to the specified range [min, max].
// This is used to prevent overflow in color calculations.
//
// Arguments:
// - value: The value to Clamp.
// - min: Minimum allowed value.
// - max: Maximum allowed value.
//
// Returns:
// - The clamped value within [min, max].
//
// @example
// clamped := Clamp(300.5, 0, 255) // Returns 255
// clamped := Clamp(-10.0, 0, 255) // Returns 0
func Clamp(value, min, max float64) float64 {
	// Check lower bound first (common case for underflow).
	if value < min {
		return min
	}
	// Check upper bound.
	if value > max {
		return max
	}
	// Value is within range.
	return value
}

// Parallel executes a function in Parallel across multiple goroutines.
// This improves performance on multi-core systems.
//
// Arguments:
// - dataSize: The size of the data to process.
// - fn: Function to execute for each partition (receives start and end indices).
//
// Returns:
// - None.
//
// @example
//
//	Parallel(height, func(start, end int) {
//	    for y := start; y < end; y++ {
//	        // Process row y
//	    }
//	})
func Parallel(dataSize int, fn func(partStart, partEnd int)) {
	// Determine number of goroutines to use.
	// We use the number of CPU cores for optimal parallelism.
	numGoroutines := runtime.NumCPU()

	// For small data sizes, parallel processing overhead isn't worth it.
	// Process serially if data is too small.
	if dataSize < numGoroutines*2 {
		fn(0, dataSize)
		return
	}

	// Calculate partition size for each goroutine.
	partSize := dataSize / numGoroutines

	// Create wait group to synchronize goroutines.
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Launch goroutines to process partitions.
	for i := 0; i < numGoroutines; i++ {
		// Calculate partition boundaries.
		partStart := i * partSize
		partEnd := partStart + partSize

		// Last partition gets any remaining data.
		if i == numGoroutines-1 {
			partEnd = dataSize
		}

		// Launch goroutine to process this partition.
		go func(start, end int) {
			// Ensure wait group is decremented when done.
			defer wg.Done()
			// Process the partition.
			fn(start, end)
		}(partStart, partEnd)
	}

	// Wait for all goroutines to complete.
	wg.Wait()
}

// EdgeMode defines how to handle coordinates that are out of bounds.
type EdgeMode string

const (
	// ClampEdgeMode clamps the pixel values to the nearest valid value.
	ClampEdgeMode EdgeMode = "clamp"
	// MirrorEdgeMode mirrors the pixel values around the edge.
	MirrorEdgeMode EdgeMode = "mirror"
	// WrapEdgeMode wraps the pixel values around the edge.
	WrapEdgeMode EdgeMode = "wrap"
)

// MapCoord maps a coordinate to a valid value based on the edge mode.
//
// Arguments:
// - coord: The coordinate to map.
// - max: The maximum value of the coordinate.
// - mode: The edge mode to use.
func MapCoord(coord, max int, mode EdgeMode) int {
	switch mode {
	case ClampEdgeMode:
		if coord < 0 {
			return 0
		} else if coord >= max {
			return max - 1
		}
		return coord
	case MirrorEdgeMode:
		for coord < 0 || coord >= max {
			if coord < 0 {
				coord = -coord - 1
			} else {
				coord = 2*max - coord - 1
			}
		}
		return coord
	case WrapEdgeMode:
		return (coord%max + max) % max
	default:
		return coord // fallback to Clamp
	}
}

// BoxBlur applies a fast box blur to an image.
// This is faster than Gaussian blur but produces lower quality results.
// Useful for real-time applications or as a pre-processing step.
//
// Arguments:
// - img: The source image to blur.
// - radius: The blur radius (box size will be 2*radius + 1).
//
// Returns:
// - A new blurred image with the same dimensions.
//
// @example
// fast := BoxBlur(img, 3)
func BoxBlur(img image.Image, radius int, mode EdgeMode) image.Image {
	// Validate radius.
	if radius <= 0 {
		// No blur needed, return a copy.
		bounds := img.Bounds()
		dst := image.NewRGBA(bounds)
		draw.Draw(dst, bounds, img, bounds.Min, draw.Src)
		return dst
	}

	// Get image bounds.
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Create intermediate and destination images.
	intermediate := image.NewRGBA(bounds)
	dst := image.NewRGBA(bounds)

	// Apply horizontal box blur.
	Parallel(height, func(partStart, partEnd int) {
		for y := partStart; y < partEnd; y++ {
			for x := 0; x < width; x++ {
				// Initialize accumulators.
				var r, g, b, a float64
				count := 0

				// Sum pixels in the box.
				for dx := -radius; dx <= radius; dx++ {
					srcX := MapCoord(x+dx, width, mode)
					// Clamp to image bounds.
					if srcX < 0 {
						srcX = 0
					} else if srcX >= width {
						srcX = width - 1
					}

					// Get pixel color.
					c := img.At(bounds.Min.X+srcX, bounds.Min.Y+y)
					srcR, srcG, srcB, srcA := c.RGBA()

					// Accumulate values.
					r += float64(srcR >> 8)
					g += float64(srcG >> 8)
					b += float64(srcB >> 8)
					a += float64(srcA >> 8)
					count++
				}

				// Calculate average.
				r /= float64(count)
				g /= float64(count)
				b /= float64(count)
				a /= float64(count)

				// Set pixel.
				intermediate.SetRGBA(x, y, color.RGBA{
					R: uint8(r + 0.5),
					G: uint8(g + 0.5),
					B: uint8(b + 0.5),
					A: uint8(a + 0.5),
				})
			}
		}
	})

	// Apply vertical box blur.
	Parallel(width, func(partStart, partEnd int) {
		for x := partStart; x < partEnd; x++ {
			for y := 0; y < height; y++ {
				// Initialize accumulators.
				var r, g, b, a float64
				count := 0

				// Sum pixels in the box.
				for dy := -radius; dy <= radius; dy++ {
					srcY := MapCoord(y+dy, height, mode)
					// Clamp to image bounds.
					if srcY < 0 {
						srcY = 0
					} else if srcY >= height {
						srcY = height - 1
					}

					// Get pixel directly (fast path).
					srcIdx := intermediate.PixOffset(x, srcY)

					// Accumulate values.
					r += float64(intermediate.Pix[srcIdx+0])
					g += float64(intermediate.Pix[srcIdx+1])
					b += float64(intermediate.Pix[srcIdx+2])
					a += float64(intermediate.Pix[srcIdx+3])
					count++
				}

				// Calculate average.
				r /= float64(count)
				g /= float64(count)
				b /= float64(count)
				a /= float64(count)

				// Set pixel directly (fast path).
				dstIdx := dst.PixOffset(x, y)
				dst.Pix[dstIdx+0] = uint8(r + 0.5)
				dst.Pix[dstIdx+1] = uint8(g + 0.5)
				dst.Pix[dstIdx+2] = uint8(b + 0.5)
				dst.Pix[dstIdx+3] = uint8(a + 0.5)
			}
		}
	})

	return dst
}

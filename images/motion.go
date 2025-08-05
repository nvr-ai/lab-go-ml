// Package images - This file contains the motion detection functionality
// using OpenCV (via gocv).
//
// The MotionSegmenter struct encapsulates a typical computer vision pipeline involving:
//  1. Background subtraction using MOG2.
//  2. Thresholding to create a binary mask of motion.
//  3. Morphological operations (e.g., dilation) to enhance regions.
//  4. Contour extraction for motion blob detection.
//
// This package is optimized for reuse across frames in a live video stream. It is
// designed for users who need reliable motion region detection without writing
// the full pipeline themselves.
//
// Pipeline Overview:
//
// ┌──────────────┐
// │ Input Frame  │
// └──────┬───────┘
// ┌────────────────────────────────────────────┐
// │ Preprocessing (resize, grayscale, denoise) │
// └──────┬─────────────────────────────────────┘
// ┌────────────────────────────┐
// │ Background Subtraction     │
// │       (MOG2)               │
// └──────┬─────────────────────┘
// ┌────────────────────────────┐
// │ Thresholding (binary mask) │
// └──────┬─────────────────────┘
// ┌────────────────────────────┐
// │ Morphology (dilate)        │
// └──────┬─────────────────────┘
// ┌────────────────────────────┐
// │ Contour Detection          │
// └──────┬─────────────────────┘
// ┌────────────────────────────┐
// │ Motion Region Output       │
// └────────────────────────────┘
//
// Usage:
//
//	seg := image.NewMotionSegmenter()
//	defer seg.Close()
//
//	for {
//	    frame := getNextFrame()
//	    contours := seg.SegmentMotion(frame)
//	    drawContours(frame, contours)
//	}
//
// Note: You must call Close() when finished to release native resources.
package images

import "gocv.io/x/gocv"

// MotionSegmenter encapsulates a typical computer vision pipeline for detecting
// motion regions in video frames. It uses background subtraction (MOG2),
// thresholding, and contour extraction to isolate regions of interest.
//
// This struct is stateful and optimized for reuse across frames in a video stream.
// Internally, it maintains OpenCV matrices and a persistent background model.
// Always call Close() when done to release native resources.
type MotionSegmenter struct {
	Input                gocv.Mat                      // Raw input frame for inspection.
	Delta                gocv.Mat                      // Foreground mask from background subtraction
	Threshold            gocv.Mat                      // Binary mask after thresholding
	Kernel               gocv.Mat                      // Morphological kernel (must be set before FillGaps)
	BackgroundSubtractor gocv.BackgroundSubtractorMOG2 // Persistent background model
}

// MotionSegmenterInput represents an image that can be passed to the MotionSegmenter.Set() method.
type MotionSegmenterInput struct {
	// The JPEG-encoded image Data.
	Data     []byte
	ReadFlag gocv.IMReadFlag
}

// NewMotionSegmenter constructs a new MotionSegmenter with initialized OpenCV matrices.
//
// It is ready to use out of the box for background subtraction and thresholding.
// However, you must initialize the kernel (via gocv.GetStructuringElement) before
// calling FillGaps() for morphological processing.
//
// Always call Close() to release memory.
func NewMotionSegmenter() *MotionSegmenter {
	return &MotionSegmenter{
		Input:                gocv.NewMat(),
		Delta:                gocv.NewMat(),
		Threshold:            gocv.NewMat(),
		Kernel:               gocv.NewMat(), // Must be explicitly initialized
		BackgroundSubtractor: gocv.NewBackgroundSubtractorMOG2(),
	}
}

// Set loads an image from a JPEG-encoded []byte and sets it as the input Mat.
//
// Arguments:
//   - input: MotionSegmenterInput containing the JPEG image data.
//
// Returns:
//   - error if decoding fails.
//
// This method decodes the JPEG bytes into a gocv.Mat using gocv.IMDecode.
// The resulting Mat is stored in the Input field for further processing.
func (m *MotionSegmenter) Set(input MotionSegmenterInput) error {
	// Decode the JPEG bytes into a Mat (color image).
	mat, err := gocv.IMDecode(input.Data, input.ReadFlag)
	if err != nil {
		return err
	}
	// Release any previous input Mat to avoid memory leaks.
	if !m.Input.Empty() {
		m.Input.Close()
	}
	m.Input = mat
	return nil
}

// SubtractBackground performs foreground segmentation using the MOG2 model.
// This generates a "delta" foreground mask that highlights motion areas.
//
// Arguments:
//   - frame: The input frame to process.
//
// Side Effect: Updates the delta field with a grayscale foreground mask.
func (m *MotionSegmenter) SubtractBackground(frame gocv.Mat) error {
	if err := m.BackgroundSubtractor.Apply(frame, &m.Delta); err != nil {
		return err
	}
	return nil
}

// ApplyThreshold converts the grayscale delta image to a binary mask.
// Pixels above the threshold become white (foreground); others become black.
//
// Arguments:
//   - threshold: Pixel intensity threshold (e.g., 25).
//   - maxVal: Maximum value to assign to foreground pixels (usually 255).
//
// Side Effect: Updates the threshold field with a binary image.
//
// Returns:
//   - The threshold used (same as input threshold).
func (m *MotionSegmenter) ApplyThreshold(threshold float32, maxVal float32) float32 {
	return gocv.Threshold(m.Delta, &m.Threshold, threshold, maxVal, gocv.ThresholdBinary)
}

// FillGaps performs a dilation operation on the binary mask to connect
// fragmented regions and fill small gaps.
//
// Improves robustness of contour detection by smoothing noisy blobs.
//
// Note: You must initialize m.kernel before calling this (e.g. via gocv.GetStructuringElement).
func (m *MotionSegmenter) FillGaps() error {
	if err := gocv.Dilate(m.Threshold, &m.Threshold, m.Kernel); err != nil {
		return err
	}
	return nil
}

// DetectContours extracts the external contours (boundaries) of connected
// regions in the binary threshold image.
//
// Uses RetrievalExternal and ChainApproxSimple for efficient region extraction.
//
// Returns:
//   - gocv.PointsVector: a vector of contours (each is a slice of gocv.Point).
func (m *MotionSegmenter) DetectContours() gocv.PointsVector {
	return gocv.FindContours(m.Threshold, gocv.RetrievalExternal, gocv.ChainApproxSimple)
}

// SegmentMotion runs the full motion segmentation pipeline:
//
//  1. Background subtraction
//  2. Thresholding
//  3. Morphological dilation
//  4. External contour detection
//
// This is a convenience method for extracting motion regions with defaults.
//
// Arguments:
//   - frame: The input frame to process.
//
// Returns:
//   - gocv.PointsVector containing all detected motion contours.
func (m *MotionSegmenter) SegmentMotion(frame gocv.Mat) gocv.PointsVector {
	m.SubtractBackground(frame)
	m.ApplyThreshold(25, 255)
	m.FillGaps()
	return m.DetectContours()
}

// DetectMotion checks if the given contours contain motion by looping through
// the contours and checking if the area is greater than the minimum area.
//
// Arguments:
//   - contours: The contours to check.
//
// Returns:
//   - bool: true if motion is detected, false otherwise.
func (m *MotionSegmenter) DetectMotion(contours gocv.PointsVector, minimumArea float64) bool {
	for i := 0; i < contours.Size(); i++ {
		area := gocv.ContourArea(contours.At(i))
		if area >= minimumArea {
			return true
		}
	}
	return false
}

// Close releases all OpenCV native resources used by the segmenter.
//
// Always call this when you're done to prevent memory leaks.
func (m *MotionSegmenter) Close() {
	m.Input.Close()
	m.Delta.Close()
	m.Threshold.Close()
	m.Kernel.Close()
	m.BackgroundSubtractor.Close()
}

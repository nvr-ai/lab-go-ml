// Package controller - Motion detection implementation using frame differencing and background subtraction
package controller

import (
	"errors"
	"fmt"
	"image"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

// MotionDetectionConfig contains configuration parameters for motion detection
type MotionDetectionConfig struct {
	// MinContourArea is the minimum area of a contour to be considered motion
	MinContourArea float64
	// DifferenceThreshold is the pixel difference threshold for motion detection
	DifferenceThreshold float64
	// BackgroundLearningRate controls how quickly the background model adapts
	BackgroundLearningRate float32
	// GaussianBlurKernelSize controls noise reduction, must be odd
	GaussianBlurKernelSize int
	// EnableBackgroundSubtraction toggles advanced background subtraction
	EnableBackgroundSubtraction bool
	// MotionHistorySeconds controls how many seconds of motion history to maintain
	MotionHistorySeconds int
}

// DefaultMotionDetectionConfig returns a default configuration for motion detection
func DefaultMotionDetectionConfig() MotionDetectionConfig {
	return MotionDetectionConfig{
		MinContourArea:              500.0,
		DifferenceThreshold:         30.0,
		BackgroundLearningRate:      0.01,
		GaussianBlurKernelSize:      21,
		EnableBackgroundSubtraction: true,
		MotionHistorySeconds:        5,
	}
}

// FrameDifferenceMotionDetector implements motion detection using frame differencing
//
// This detector compares consecutive frames to identify areas of change that indicate motion.
// It maintains a background model and tracks motion history for temporal consistency.
type FrameDifferenceMotionDetector struct {
	config         MotionDetectionConfig
	previousFrame  gocv.Mat
	backgroundMOG2 gocv.BackgroundSubtractorMOG2
	motionHistory  []float64
	frameCount     int64
	mu             sync.RWMutex
	initialized    bool
}

// NewFrameDifferenceMotionDetector creates a new frame differencing motion detector
//
// Arguments:
//   - config: Configuration parameters for motion detection
//
// Returns:
//   - *FrameDifferenceMotionDetector: The initialized motion detector
//
// @example
// config := DefaultMotionDetectionConfig()
// detector := NewFrameDifferenceMotionDetector(config)
// defer detector.Close()
func NewFrameDifferenceMotionDetector(config MotionDetectionConfig) *FrameDifferenceMotionDetector {
	detector := &FrameDifferenceMotionDetector{
		config:        config,
		previousFrame: gocv.NewMat(),
		motionHistory: make([]float64, 0, config.MotionHistorySeconds*30), // Assume 30 FPS max
		frameCount:    0,
		initialized:   false,
	}

	// Initialize MOG2 background subtractor if enabled
	if config.EnableBackgroundSubtraction {
		detector.backgroundMOG2 = gocv.NewBackgroundSubtractorMOG2WithParams(
			500,                              // History length
			16.0,                             // Variance threshold
			false,                            // Detect shadows
		)
	}

	return detector
}

// DetectMotion detects motion in the given frame and returns a motion score
//
// The motion score ranges from 0.0 (no motion) to 1.0 (maximum motion).
// This implementation uses frame differencing and optional background subtraction
// to identify motion areas and calculate an overall motion score.
//
// Arguments:
//   - frame: The video frame to analyze for motion
//
// Returns:
//   - float64: Motion score between 0.0 and 1.0
//   - error: An error if motion detection fails
//
// @example
// detector := NewFrameDifferenceMotionDetector(DefaultMotionDetectionConfig())
// score, err := detector.DetectMotion(frame)
// if err != nil {
//     log.Printf("Motion detection failed: %v", err)
// }
// fmt.Printf("Motion score: %.3f\n", score)
func (fmd *FrameDifferenceMotionDetector) DetectMotion(frame Frame) (float64, error) {
	fmd.mu.Lock()
	defer fmd.mu.Unlock()

	// Convert image.Image to gocv.Mat
	currentMat, err := fmd.imageToMat(frame.Image)
	if err != nil {
		return 0.0, fmt.Errorf("failed to convert image to Mat: %w", err)
	}
	defer currentMat.Close()

	// Convert to grayscale for processing
	grayMat := gocv.NewMat()
	defer grayMat.Close()
	gocv.CvtColor(currentMat, &grayMat, gocv.ColorBGRToGray)

	// Apply Gaussian blur to reduce noise
	blurredMat := gocv.NewMat()
	defer blurredMat.Close()
	gocv.GaussianBlur(grayMat, &blurredMat, 
		image.Pt(fmd.config.GaussianBlurKernelSize, fmd.config.GaussianBlurKernelSize),
		0, 0, gocv.BorderDefault)

	fmd.frameCount++

	// Initialize on first frame
	if !fmd.initialized {
		blurredMat.CopyTo(&fmd.previousFrame)
		fmd.initialized = true
		return 0.0, nil
	}

	var motionScore float64

	if fmd.config.EnableBackgroundSubtraction {
		motionScore, err = fmd.detectMotionBackgroundSubtraction(blurredMat)
	} else {
		motionScore, err = fmd.detectMotionFrameDifference(blurredMat)
	}

	if err != nil {
		return 0.0, err
	}

	// Update motion history
	fmd.updateMotionHistory(motionScore)

	// Update previous frame for next comparison
	blurredMat.CopyTo(&fmd.previousFrame)

	// Apply temporal smoothing using motion history
	smoothedScore := fmd.getSmoothedMotionScore()

	return smoothedScore, nil
}

// detectMotionFrameDifference implements basic frame differencing motion detection
func (fmd *FrameDifferenceMotionDetector) detectMotionFrameDifference(currentFrame gocv.Mat) (float64, error) {
	// Calculate absolute difference between current and previous frame
	diffMat := gocv.NewMat()
	defer diffMat.Close()
	gocv.AbsDiff(currentFrame, fmd.previousFrame, &diffMat)

	// Apply threshold to get binary motion mask
	thresholdMat := gocv.NewMat()
	defer thresholdMat.Close()
	gocv.Threshold(diffMat, &thresholdMat, float32(fmd.config.DifferenceThreshold), 255, gocv.ThresholdBinary)

	// Find contours in the motion mask
	contours := gocv.FindContours(thresholdMat, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	// Calculate motion score based on contour areas
	totalMotionArea := 0.0
	frameArea := float64(currentFrame.Rows() * currentFrame.Cols())

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)
		contour.Close()

		if area >= fmd.config.MinContourArea {
			totalMotionArea += area
		}
	}

	// Normalize motion score to [0, 1] range
	motionScore := math.Min(totalMotionArea/frameArea, 1.0)
	return motionScore, nil
}

// detectMotionBackgroundSubtraction implements advanced background subtraction
func (fmd *FrameDifferenceMotionDetector) detectMotionBackgroundSubtraction(currentFrame gocv.Mat) (float64, error) {
	// Apply background subtraction
	fgMask := gocv.NewMat()
	defer fgMask.Close()
	
	fmd.backgroundMOG2.Apply(currentFrame, &fgMask)

	// Morphological operations to clean up the mask
	kernel := gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(5, 5))
	defer kernel.Close()

	cleanMask := gocv.NewMat()
	defer cleanMask.Close()
	
	// Opening to remove noise
	gocv.MorphologyEx(fgMask, &cleanMask, gocv.MorphOpen, kernel)
	
	// Closing to fill holes
	gocv.MorphologyEx(cleanMask, &cleanMask, gocv.MorphClose, kernel)

	// Find contours in the cleaned mask
	contours := gocv.FindContours(cleanMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	// Calculate motion score based on contour areas and count
	totalMotionArea := 0.0
	motionObjectCount := 0
	frameArea := float64(currentFrame.Rows() * currentFrame.Cols())

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)
		contour.Close()

		if area >= fmd.config.MinContourArea {
			totalMotionArea += area
			motionObjectCount++
		}
	}

	// Combine area-based and count-based scoring
	areaScore := math.Min(totalMotionArea/frameArea, 1.0)
	countScore := math.Min(float64(motionObjectCount)/10.0, 1.0) // Normalize to max 10 objects
	
	// Weighted combination (70% area, 30% count)
	motionScore := 0.7*areaScore + 0.3*countScore
	return math.Min(motionScore, 1.0), nil
}

// updateMotionHistory adds a new motion score to the history buffer
func (fmd *FrameDifferenceMotionDetector) updateMotionHistory(score float64) {
	maxHistorySize := fmd.config.MotionHistorySeconds * 30 // Assume 30 FPS max
	
	fmd.motionHistory = append(fmd.motionHistory, score)
	
	// Keep only the most recent scores
	if len(fmd.motionHistory) > maxHistorySize {
		fmd.motionHistory = fmd.motionHistory[len(fmd.motionHistory)-maxHistorySize:]
	}
}

// getSmoothedMotionScore calculates a smoothed motion score using recent history
func (fmd *FrameDifferenceMotionDetector) getSmoothedMotionScore() float64 {
	if len(fmd.motionHistory) == 0 {
		return 0.0
	}

	// Calculate weighted average with recent frames having higher weight
	totalScore := 0.0
	totalWeight := 0.0
	
	for i, score := range fmd.motionHistory {
		// Linear weighting: more recent frames get higher weight
		weight := float64(i+1) / float64(len(fmd.motionHistory))
		totalScore += score * weight
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0.0
	}

	return totalScore / totalWeight
}

// imageToMat converts an image.Image to a gocv.Mat
func (fmd *FrameDifferenceMotionDetector) imageToMat(img image.Image) (gocv.Mat, error) {
	if img == nil {
		return gocv.NewMat(), errors.New("input image is nil")
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Create a new Mat with the appropriate size
	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)

	// Convert image.Image to Mat data
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Convert from 16-bit to 8-bit and BGR format for OpenCV
			mat.SetUCharAt(y-bounds.Min.Y, (x-bounds.Min.X)*3+0, uint8(b>>8))   // B
			mat.SetUCharAt(y-bounds.Min.Y, (x-bounds.Min.X)*3+1, uint8(g>>8))   // G
			mat.SetUCharAt(y-bounds.Min.Y, (x-bounds.Min.X)*3+2, uint8(r>>8))   // R
		}
	}

	return mat, nil
}

// GetMotionHistory returns the recent motion history for analysis
//
// Returns:
//   - []float64: Slice of recent motion scores
//
// @example
// detector := NewFrameDifferenceMotionDetector(config)
// history := detector.GetMotionHistory()
// fmt.Printf("Recent motion scores: %v\n", history)
func (fmd *FrameDifferenceMotionDetector) GetMotionHistory() []float64 {
	fmd.mu.RLock()
	defer fmd.mu.RUnlock()
	
	// Return a copy to prevent external modification
	history := make([]float64, len(fmd.motionHistory))
	copy(history, fmd.motionHistory)
	return history
}

// GetFrameCount returns the total number of frames processed
//
// Returns:
//   - int64: Number of frames processed since initialization
func (fmd *FrameDifferenceMotionDetector) GetFrameCount() int64 {
	fmd.mu.RLock()
	defer fmd.mu.RUnlock()
	return fmd.frameCount
}

// Reset clears the motion detection state and history
//
// This method reinitializes the detector, clearing all history and background models.
// Use this when switching between different video streams or after long pauses.
func (fmd *FrameDifferenceMotionDetector) Reset() {
	fmd.mu.Lock()
	defer fmd.mu.Unlock()
	
	fmd.previousFrame.Close()
	fmd.previousFrame = gocv.NewMat()
	fmd.motionHistory = fmd.motionHistory[:0]
	fmd.frameCount = 0
	fmd.initialized = false
	
	// Reset background subtractor if enabled
	if fmd.config.EnableBackgroundSubtraction {
		fmd.backgroundMOG2.Close()
		fmd.backgroundMOG2 = gocv.NewBackgroundSubtractorMOG2WithParams(
			500,                           // History length
			16.0,                          // Variance threshold
			false,                         // Detect shadows
		)
	}
}

// Close releases all resources associated with the motion detector
//
// This method must be called when the detector is no longer needed to prevent memory leaks.
// It closes all OpenCV Mat objects and background subtractors.
func (fmd *FrameDifferenceMotionDetector) Close() {
	fmd.mu.Lock()
	defer fmd.mu.Unlock()
	
	if !fmd.previousFrame.Empty() {
		fmd.previousFrame.Close()
	}
	
	if fmd.config.EnableBackgroundSubtraction {
		fmd.backgroundMOG2.Close()
	}
}

// GetConfig returns the current motion detection configuration
//
// Returns:
//   - MotionDetectionConfig: Current configuration parameters
func (fmd *FrameDifferenceMotionDetector) GetConfig() MotionDetectionConfig {
	fmd.mu.RLock()
	defer fmd.mu.RUnlock()
	return fmd.config
}

// UpdateConfig updates the motion detection configuration
//
// Arguments:
//   - config: New configuration parameters
//
// Note: Changes to background subtraction settings require a Reset() call to take effect.
func (fmd *FrameDifferenceMotionDetector) UpdateConfig(config MotionDetectionConfig) {
	fmd.mu.Lock()
	defer fmd.mu.Unlock()
	fmd.config = config
}
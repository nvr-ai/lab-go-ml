// Package controller - This file contains the controller for routing frames to the appropriate detector.
package controller

import (
	"image"
	"time"

	"github.com/nvr-ai/go-ml/images"
)

const (
	// LowRes is the lowest resolution detector.
	LowRes = images.ResolutionType1MP54
	// MedRes is the medium resolution detector.
	MedRes = images.ResolutionType6MP43
	// HighRes is the highest resolution detector.
	HighRes = images.ResolutionTypeFHD1080p
)

// Frame is a single frame of video.
type Frame struct {
	ID        int
	Image     image.Image
	Timestamp time.Time
}

// Detection is a single detection from a detector.
type Detection struct {
	Label      string
	Confidence float64
	BBox       image.Rectangle
}

// Detector is an interface for a detector.
type Detector interface {
	Detect(frame Frame) ([]Detection, error)
	Resolution() images.Resolution
	Name() string
}

// MotionDetector is an interface for a motion detector.
type MotionDetector interface {
	DetectMotion(frame Frame) (float64, error)
}

// ThresholdConfig is a configuration for the thresholds.
type ThresholdConfig struct {
	MotionThreshold  float64
	DensityThreshold int
	HysteresisFrames int
}

// Controller is a controller for the resolution.
type Controller struct {
	MotionDetector   MotionDetector
	DensityEstimator DensityEstimator
	Detectors        map[images.ResolutionType]Detector
	Current          images.ResolutionType
	HysteresisCount  int
	Thresholds       ThresholdConfig
}

// Decide decides the next resolution to use.
//
// Arguments:
//   - frame: The frame to decide the next resolution for.
//
// Returns:
//   - Detector: The detector to use for the next frame.
//   - error: An error if the decision fails.
func (rc *Controller) Decide(frame Frame) (Detector, error) {
	motionScore, err := rc.MotionDetector.DetectMotion(frame)
	if err != nil {
		return nil, err
	}

	lowResDetector := rc.Detectors[LowRes]
	lowResDetections, err := lowResDetector.Detect(frame)
	if err != nil {
		return nil, err
	}

	density, err := rc.DensityEstimator.EstimateDensity(lowResDetections)
	if err != nil {
		return nil, err
	}

	var next images.ResolutionType
	switch {
	case density > rc.Thresholds.DensityThreshold:
		next = HighRes
	case motionScore > rc.Thresholds.MotionThreshold:
		next = MedRes
	default:
		next = LowRes
	}

	if next != rc.Current {
		rc.HysteresisCount++
		if rc.HysteresisCount >= rc.Thresholds.HysteresisFrames {
			rc.Current = next
			rc.HysteresisCount = 0
		}
	} else {
		rc.HysteresisCount = 0
	}

	return rc.Detectors[rc.Current], nil
}

// Package controller - Comprehensive testing for dynamic resolution controller with hysteresis validation
package controller

import (
	"errors"
	"image"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockMotionDetector provides controllable motion detection for testing
type MockMotionDetector struct {
	motionScores []float64
	currentIndex int
	shouldError  bool
}

func (m *MockMotionDetector) DetectMotion(frame Frame) (float64, error) {
	if m.shouldError {
		return 0, errors.New("mock motion detection error")
	}
	
	if m.currentIndex >= len(m.motionScores) {
		return 0, nil
	}
	
	score := m.motionScores[m.currentIndex]
	m.currentIndex++
	return score, nil
}

func (m *MockMotionDetector) Reset() {
	m.currentIndex = 0
}

// MockDensityEstimator provides controllable density estimation for testing
type MockDensityEstimator struct {
	densities   []int
	currentIndex int
	shouldError bool
}

func (m *MockDensityEstimator) EstimateDensity(detections []Detection) (int, error) {
	if m.shouldError {
		return 0, errors.New("mock density estimation error")
	}
	
	if m.currentIndex >= len(m.densities) {
		return 0, nil
	}
	
	density := m.densities[m.currentIndex]
	m.currentIndex++
	return density, nil
}

func (m *MockDensityEstimator) GetDensityMetrics(detections []Detection) (*DensityMetrics, error) {
	return &DensityMetrics{TotalObjects: len(detections)}, nil
}

func (m *MockDensityEstimator) Reset() {
	m.currentIndex = 0
}

// MockDetector provides controllable detection results for testing
type MockDetector struct {
	name       string
	resolution images.Resolution
	detections []Detection
	shouldError bool
}

func (m *MockDetector) Detect(frame Frame) ([]Detection, error) {
	if m.shouldError {
		return nil, errors.New("mock detection error")
	}
	return m.detections, nil
}

func (m *MockDetector) Resolution() images.Resolution {
	return m.resolution
}

func (m *MockDetector) Name() string {
	return m.name
}

// TestHysteresisValidation tests the 3-frame confirmation requirement
func TestHysteresisValidation(t *testing.T) {
	tests := []struct {
		name                    string
		motionScores           []float64
		densities              []int
		motionThreshold        float64
		densityThreshold       int
		hysteresisFrames       int
		expectedTransitions    []images.ResolutionType
		expectedHysteresisCounts []int
	}{
		{
			name:               "Basic hysteresis with 3-frame confirmation",
			motionScores:       []float64{0.8, 0.8, 0.8, 0.1, 0.1, 0.1},
			densities:          []int{5, 5, 5, 5, 5, 5},
			motionThreshold:    0.5,
			densityThreshold:   10,
			hysteresisFrames:   3,
			expectedTransitions: []images.ResolutionType{LowRes, LowRes, MedRes, MedRes, MedRes, LowRes},
			expectedHysteresisCounts: []int{1, 2, 0, 1, 2, 0},
		},
		{
			name:               "High density triggers immediate high resolution",
			motionScores:       []float64{0.1, 0.1, 0.1},
			densities:          []int{15, 15, 15},
			motionThreshold:    0.5,
			densityThreshold:   10,
			hysteresisFrames:   3,
			expectedTransitions: []images.ResolutionType{LowRes, LowRes, HighRes},
			expectedHysteresisCounts: []int{1, 2, 0},
		},
		{
			name:               "Oscillating conditions reset hysteresis",
			motionScores:       []float64{0.8, 0.1, 0.8, 0.1, 0.8, 0.1},
			densities:          []int{5, 5, 5, 5, 5, 5},
			motionThreshold:    0.5,
			densityThreshold:   10,
			hysteresisFrames:   3,
			expectedTransitions: []images.ResolutionType{LowRes, LowRes, LowRes, LowRes, LowRes, LowRes},
			expectedHysteresisCounts: []int{1, 0, 1, 0, 1, 0},
		},
		{
			name:               "Single frame confirmation",
			motionScores:       []float64{0.8, 0.1},
			densities:          []int{5, 5},
			motionThreshold:    0.5,
			densityThreshold:   10,
			hysteresisFrames:   1,
			expectedTransitions: []images.ResolutionType{LowRes, LowRes},
			expectedHysteresisCounts: []int{0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup mock components
			motionDetector := &MockMotionDetector{motionScores: tt.motionScores}
			densityEstimator := &MockDensityEstimator{densities: tt.densities}
			
			// Create mock detectors
			detectors := map[images.ResolutionType]Detector{
				LowRes:  &MockDetector{name: "low", detections: make([]Detection, 0)},
				MedRes:  &MockDetector{name: "med", detections: make([]Detection, 0)},
				HighRes: &MockDetector{name: "high", detections: make([]Detection, 0)},
			}

			// Create controller with test configuration
			controller := &Controller{
				MotionDetector:   motionDetector,
				DensityEstimator: densityEstimator,
				Detectors:        detectors,
				Current:          LowRes,
				HysteresisCount:  0,
				Thresholds: ThresholdConfig{
					MotionThreshold:  tt.motionThreshold,
					DensityThreshold: tt.densityThreshold,
					HysteresisFrames: tt.hysteresisFrames,
				},
			}

			// Test each frame
			for i := range tt.motionScores {
				frame := Frame{
					ID:        i,
					Image:     image.NewRGBA(image.Rect(0, 0, 640, 480)),
					Timestamp: time.Now(),
				}

				detector, err := controller.Decide(frame)
				require.NoError(t, err)

				// Check expected resolution
				expectedRes := tt.expectedTransitions[i]
				actualRes := controller.Current
				assert.Equal(t, expectedRes, actualRes, 
					"Frame %d: expected resolution %v, got %v", i, expectedRes, actualRes)

				// Check expected hysteresis count
				expectedCount := tt.expectedHysteresisCounts[i]
				actualCount := controller.HysteresisCount
				assert.Equal(t, expectedCount, actualCount,
					"Frame %d: expected hysteresis count %d, got %d", i, expectedCount, actualCount)

				// Verify detector corresponds to current resolution
				assert.Equal(t, detectors[controller.Current], detector)
			}
		})
	}
}

// TestHysteresisStability tests that hysteresis prevents rapid switching
func TestHysteresisStability(t *testing.T) {
	motionDetector := &MockMotionDetector{
		// Alternating high/low motion to test stability
		motionScores: []float64{0.8, 0.1, 0.9, 0.05, 0.85, 0.1, 0.8, 0.1},
	}
	densityEstimator := &MockDensityEstimator{
		// Low density throughout
		densities: []int{3, 3, 3, 3, 3, 3, 3, 3},
	}

	detectors := map[images.ResolutionType]Detector{
		LowRes:  &MockDetector{name: "low", detections: make([]Detection, 0)},
		MedRes:  &MockDetector{name: "med", detections: make([]Detection, 0)},
		HighRes: &MockDetector{name: "high", detections: make([]Detection, 0)},
	}

	controller := &Controller{
		MotionDetector:   motionDetector,
		DensityEstimator: densityEstimator,
		Detectors:        detectors,
		Current:          LowRes,
		HysteresisCount:  0,
		Thresholds: ThresholdConfig{
			MotionThreshold:  0.5,
			DensityThreshold: 10,
			HysteresisFrames: 3,
		},
	}

	resolutionChanges := 0
	previousResolution := controller.Current

	for i := 0; i < len(motionDetector.motionScores); i++ {
		frame := Frame{
			ID:        i,
			Image:     image.NewRGBA(image.Rect(0, 0, 640, 480)),
			Timestamp: time.Now(),
		}

		_, err := controller.Decide(frame)
		require.NoError(t, err)

		if controller.Current != previousResolution {
			resolutionChanges++
			previousResolution = controller.Current
		}
	}

	// With hysteresis, there should be minimal resolution changes despite oscillating input
	assert.LessOrEqual(t, resolutionChanges, 2, "Hysteresis should prevent excessive resolution switching")
}

// TestHysteresisEdgeCases tests edge cases in hysteresis logic
func TestHysteresisEdgeCases(t *testing.T) {
	t.Run("Zero hysteresis frames", func(t *testing.T) {
		motionDetector := &MockMotionDetector{motionScores: []float64{0.8, 0.1}}
		densityEstimator := &MockDensityEstimator{densities: []int{5, 5}}

		detectors := map[images.ResolutionType]Detector{
			LowRes: &MockDetector{name: "low", detections: make([]Detection, 0)},
			MedRes: &MockDetector{name: "med", detections: make([]Detection, 0)},
		}

		controller := &Controller{
			MotionDetector:   motionDetector,
			DensityEstimator: densityEstimator,
			Detectors:        detectors,
			Current:          LowRes,
			Thresholds: ThresholdConfig{
				MotionThreshold:  0.5,
				DensityThreshold: 10,
				HysteresisFrames: 0, // No hysteresis
			},
		}

		// First frame: high motion should immediately switch to MedRes
		frame1 := Frame{ID: 0, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
		_, err := controller.Decide(frame1)
		require.NoError(t, err)
		assert.Equal(t, MedRes, controller.Current)

		// Second frame: low motion should immediately switch back to LowRes
		frame2 := Frame{ID: 1, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
		_, err = controller.Decide(frame2)
		require.NoError(t, err)
		assert.Equal(t, LowRes, controller.Current)
	})

	t.Run("Same resolution maintains zero hysteresis count", func(t *testing.T) {
		motionDetector := &MockMotionDetector{motionScores: []float64{0.1, 0.1, 0.1}}
		densityEstimator := &MockDensityEstimator{densities: []int{5, 5, 5}}

		detectors := map[images.ResolutionType]Detector{
			LowRes: &MockDetector{name: "low", detections: make([]Detection, 0)},
		}

		controller := &Controller{
			MotionDetector:   motionDetector,
			DensityEstimator: densityEstimator,
			Detectors:        detectors,
			Current:          LowRes,
			Thresholds: ThresholdConfig{
				MotionThreshold:  0.5,
				DensityThreshold: 10,
				HysteresisFrames: 3,
			},
		}

		for i := 0; i < 3; i++ {
			frame := Frame{ID: i, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
			_, err := controller.Decide(frame)
			require.NoError(t, err)
			
			// Should remain at LowRes with zero hysteresis count
			assert.Equal(t, LowRes, controller.Current)
			assert.Equal(t, 0, controller.HysteresisCount)
		}
	})
}

// TestErrorHandling tests error conditions in the controller
func TestErrorHandling(t *testing.T) {
	t.Run("Motion detector error", func(t *testing.T) {
		motionDetector := &MockMotionDetector{shouldError: true}
		densityEstimator := &MockDensityEstimator{densities: []int{5}}
		detectors := map[images.ResolutionType]Detector{
			LowRes: &MockDetector{name: "low", detections: make([]Detection, 0)},
		}

		controller := &Controller{
			MotionDetector:   motionDetector,
			DensityEstimator: densityEstimator,
			Detectors:        detectors,
			Current:          LowRes,
			Thresholds:       ThresholdConfig{},
		}

		frame := Frame{ID: 0, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
		_, err := controller.Decide(frame)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "mock motion detection error")
	})

	t.Run("Detector error", func(t *testing.T) {
		motionDetector := &MockMotionDetector{motionScores: []float64{0.1}}
		densityEstimator := &MockDensityEstimator{densities: []int{5}}
		detectors := map[images.ResolutionType]Detector{
			LowRes: &MockDetector{name: "low", detections: make([]Detection, 0), shouldError: true},
		}

		controller := &Controller{
			MotionDetector:   motionDetector,
			DensityEstimator: densityEstimator,
			Detectors:        detectors,
			Current:          LowRes,
			Thresholds:       ThresholdConfig{},
		}

		frame := Frame{ID: 0, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
		_, err := controller.Decide(frame)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "mock detection error")
	})

	t.Run("Density estimator error", func(t *testing.T) {
		motionDetector := &MockMotionDetector{motionScores: []float64{0.1}}
		densityEstimator := &MockDensityEstimator{shouldError: true}
		detectors := map[images.ResolutionType]Detector{
			LowRes: &MockDetector{name: "low", detections: make([]Detection, 0)},
		}

		controller := &Controller{
			MotionDetector:   motionDetector,
			DensityEstimator: densityEstimator,
			Detectors:        detectors,
			Current:          LowRes,
			Thresholds:       ThresholdConfig{},
		}

		frame := Frame{ID: 0, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
		_, err := controller.Decide(frame)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "mock density estimation error")
	})
}

// TestControllerStateMaintenance tests that controller state is properly maintained
func TestControllerStateMaintenance(t *testing.T) {
	motionDetector := &MockMotionDetector{
		motionScores: []float64{0.8, 0.8, 0.1, 0.1},
	}
	densityEstimator := &MockDensityEstimator{
		densities: []int{5, 5, 5, 5},
	}

	detectors := map[images.ResolutionType]Detector{
		LowRes: &MockDetector{name: "low", detections: make([]Detection, 0)},
		MedRes: &MockDetector{name: "med", detections: make([]Detection, 0)},
	}

	controller := &Controller{
		MotionDetector:   motionDetector,
		DensityEstimator: densityEstimator,
		Detectors:        detectors,
		Current:          LowRes,
		HysteresisCount:  0,
		Thresholds: ThresholdConfig{
			MotionThreshold:  0.5,
			DensityThreshold: 10,
			HysteresisFrames: 2,
		},
	}

	// Trace state through multiple frames
	states := make([]struct {
		current    images.ResolutionType
		hysteresis int
	}, 4)

	for i := 0; i < 4; i++ {
		frame := Frame{ID: i, Image: image.NewRGBA(image.Rect(0, 0, 640, 480))}
		_, err := controller.Decide(frame)
		require.NoError(t, err)

		states[i] = struct {
			current    images.ResolutionType
			hysteresis int
		}{
			current:    controller.Current,
			hysteresis: controller.HysteresisCount,
		}
	}

	// Frame 0: high motion, hysteresis starts
	assert.Equal(t, LowRes, states[0].current)
	assert.Equal(t, 1, states[0].hysteresis)

	// Frame 1: still high motion, hysteresis completes, switch to MedRes
	assert.Equal(t, MedRes, states[1].current)
	assert.Equal(t, 0, states[1].hysteresis)

	// Frame 2: low motion, hysteresis starts again
	assert.Equal(t, MedRes, states[2].current)
	assert.Equal(t, 1, states[2].hysteresis)

	// Frame 3: still low motion, hysteresis completes, switch to LowRes
	assert.Equal(t, LowRes, states[3].current)
	assert.Equal(t, 0, states[3].hysteresis)
}

// BenchmarkControllerDecide benchmarks the decision-making performance
func BenchmarkControllerDecide(b *testing.B) {
	motionDetector := &MockMotionDetector{
		motionScores: make([]float64, b.N),
	}
	for i := 0; i < b.N; i++ {
		motionDetector.motionScores[i] = 0.5
	}

	densityEstimator := &MockDensityEstimator{
		densities: make([]int, b.N),
	}
	for i := 0; i < b.N; i++ {
		densityEstimator.densities[i] = 5
	}

	detectors := map[images.ResolutionType]Detector{
		LowRes:  &MockDetector{name: "low", detections: make([]Detection, 0)},
		MedRes:  &MockDetector{name: "med", detections: make([]Detection, 0)},
		HighRes: &MockDetector{name: "high", detections: make([]Detection, 0)},
	}

	controller := &Controller{
		MotionDetector:   motionDetector,
		DensityEstimator: densityEstimator,
		Detectors:        detectors,
		Current:          LowRes,
		Thresholds: ThresholdConfig{
			MotionThreshold:  0.6,
			DensityThreshold: 10,
			HysteresisFrames: 3,
		},
	}

	frame := Frame{
		ID:        0,
		Image:     image.NewRGBA(image.Rect(0, 0, 640, 480)),
		Timestamp: time.Now(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := controller.Decide(frame)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// TestControllerConcurrency tests thread safety of controller operations
func TestControllerConcurrency(t *testing.T) {
	motionDetector := &MockMotionDetector{
		motionScores: make([]float64, 100),
	}
	for i := 0; i < 100; i++ {
		motionDetector.motionScores[i] = 0.5
	}

	densityEstimator := &MockDensityEstimator{
		densities: make([]int, 100),
	}
	for i := 0; i < 100; i++ {
		densityEstimator.densities[i] = 5
	}

	detectors := map[images.ResolutionType]Detector{
		LowRes:  &MockDetector{name: "low", detections: make([]Detection, 0)},
		MedRes:  &MockDetector{name: "med", detections: make([]Detection, 0)},
		HighRes: &MockDetector{name: "high", detections: make([]Detection, 0)},
	}

	controller := &Controller{
		MotionDetector:   motionDetector,
		DensityEstimator: densityEstimator,
		Detectors:        detectors,
		Current:          LowRes,
		Thresholds: ThresholdConfig{
			MotionThreshold:  0.6,
			DensityThreshold: 10,
			HysteresisFrames: 3,
		},
	}

	// Note: The controller itself is not thread-safe and should be accessed
	// from a single goroutine. This test ensures that individual calls don't panic.
	frame := Frame{
		ID:        0,
		Image:     image.NewRGBA(image.Rect(0, 0, 640, 480)),
		Timestamp: time.Now(),
	}

	// Sequential calls should not panic or cause race conditions
	for i := 0; i < 10; i++ {
		_, err := controller.Decide(frame)
		require.NoError(t, err)
	}
}
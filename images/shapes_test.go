package images

import (
	"image"
	"math"
	"testing"
)

// TestIoU_Correctness validates the IoU implementation against known test cases
func TestIoU_Correctness(t *testing.T) {
	tests := []struct {
		name     string
		r1       Rect
		r2       Rect
		expected float32
		epsilon  float32
	}{
		{
			name:     "Identical rectangles",
			r1:       Rect{0, 0, 100, 100},
			r2:       Rect{0, 0, 100, 100},
			expected: 1.0,
			epsilon:  0.001,
		},
		{
			name:     "No overlap",
			r1:       Rect{0, 0, 100, 100},
			r2:       Rect{200, 200, 300, 300},
			expected: 0.0,
			epsilon:  0.001,
		},
		{
			name:     "Touching edges",
			r1:       Rect{0, 0, 100, 100},
			r2:       Rect{100, 0, 200, 100},
			expected: 0.0,
			epsilon:  0.001,
		},
		{
			name:     "Half overlap",
			r1:       Rect{0, 0, 100, 100},
			r2:       Rect{50, 50, 150, 150},
			expected: 0.142857, // intersection=2500, union=10000+10000-2500=17500, iou=2500/17500=1/7≈0.142857
			epsilon:  0.001,
		},
		{
			name:     "Small overlap",
			r1:       Rect{0, 0, 100, 100},
			r2:       Rect{90, 90, 190, 190},
			expected: 0.005025, // intersection=100, union=10000+10000-100=19900, iou=100/19900≈0.00502
			epsilon:  0.001,
		},
		{
			name:     "One inside other",
			r1:       Rect{0, 0, 100, 100},
			r2:       Rect{25, 25, 75, 75},
			expected: 0.25, // intersection=2500, union=10000, iou=2500/10000=0.25
			epsilon:  0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateIoU(tt.r1, tt.r2)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("IoU() = %v, expected %v (±%v)", result, tt.expected, tt.epsilon)
			}

			// Test symmetry: IoU(A, B) should equal IoU(B, A)
			reverse := CalculateIoU(tt.r2, tt.r1)
			if math.Abs(float64(result-reverse)) > float64(tt.epsilon) {
				t.Errorf("IoU not symmetric: IoU(A,B)=%v != IoU(B,A)=%v", result, reverse)
			}
		})
	}
}

// TestIoU_vs_ImageRectangle compares our implementation against image.Rectangle
func TestIoU_vs_ImageRectangle(t *testing.T) {
	testCases := []struct {
		name string
		r1   Rect
		r2   Rect
	}{
		{"No overlap", Rect{0, 0, 100, 100}, Rect{200, 200, 300, 300}},
		{"Partial overlap", Rect{0, 0, 100, 100}, Rect{50, 50, 150, 150}},
		{"Full overlap", Rect{50, 50, 150, 150}, Rect{50, 50, 150, 150}},
		{"One inside other", Rect{0, 0, 100, 100}, Rect{25, 25, 75, 75}},
		{"Large boxes", Rect{0, 0, 1920, 1080}, Rect{960, 540, 1920, 1080}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Our implementation
			customResult := CalculateIoU(tc.r1, tc.r2)

			// image.Rectangle implementation
			ir1 := image.Rect(tc.r1.X1, tc.r1.Y1, tc.r1.X2, tc.r1.Y2)
			ir2 := image.Rect(tc.r2.X1, tc.r2.Y1, tc.r2.X2, tc.r2.Y2)
			imageResult := imageRectAngleIoU(ir1, ir2)

			// Results should be identical (within floating point precision)
			if math.Abs(float64(customResult-imageResult)) > 0.0001 {
				t.Errorf("Results differ: custom=%v, image.Rectangle=%v", customResult, imageResult)
			}
		})
	}
}

// imageRectangleIoU implements IoU using Go's standard library image.Rectangle
func imageRectAngleIoU(r1, r2 image.Rectangle) float32 {
	intersect := r1.Intersect(r2)
	if intersect.Empty() {
		return 0.0
	}

	intersectArea := intersect.Dx() * intersect.Dy()
	r1Area := r1.Dx() * r1.Dy()
	r2Area := r2.Dx() * r2.Dy()
	union := r1Area + r2Area - intersectArea

	return float32(intersectArea) / float32(union)
}

// TestIoU_EdgeCases tests edge cases and boundary conditions
func TestIoU_EdgeCases(t *testing.T) {
	tests := []struct {
		name string
		r1   Rect
		r2   Rect
	}{
		{"Zero area rectangle 1", Rect{0, 0, 0, 0}, Rect{0, 0, 100, 100}},
		{"Zero area rectangle 2", Rect{0, 0, 100, 100}, Rect{50, 50, 50, 50}},
		{"Both zero area", Rect{0, 0, 0, 0}, Rect{10, 10, 10, 10}},
		{"Negative coordinates", Rect{-100, -100, 0, 0}, Rect{-50, -50, 50, 50}},
		{"Single pixel", Rect{0, 0, 1, 1}, Rect{0, 0, 1, 1}},
		{"Very large coordinates", Rect{0, 0, 999999, 999999}, Rect{500000, 500000, 999999, 999999}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Should not panic and should return valid result
			result := CalculateIoU(tt.r1, tt.r2)
			if result < 0.0 || result > 1.0 {
				t.Errorf("IoU result %v is outside valid range [0.0, 1.0]", result)
			}

			// Should not panic with reverse order
			reverseResult := CalculateIoU(tt.r2, tt.r1)
			if reverseResult < 0.0 || reverseResult > 1.0 {
				t.Errorf("Reverse IoU result %v is outside valid range [0.0, 1.0]", reverseResult)
			}
		})
	}
}

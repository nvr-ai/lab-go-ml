package test

import (
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

// MockFrameGenerator creates deterministic test frames for idempotent testing.
//
// Arguments:
// - None.
//
// Returns:
// - A generator for creating test frames with controlled motion patterns.
//
// @example
// gen := NewMockFrameGenerator(640, 480)
// frame := gen.GenerateStaticFrame()
// defer frame.Close()
type MockFrameGenerator struct {
	width  int
	height int
	seed   int64
}

// NewMockFrameGenerator creates a new frame generator with specified dimensions.
//
// Arguments:
// - width: Frame width in pixels.
// - height: Frame height in pixels.
//
// Returns:
// - A configured MockFrameGenerator instance.
//
// @example
// gen := NewMockFrameGenerator(1920, 1080)
func NewMockFrameGenerator(width, height int) *MockFrameGenerator {
	return &MockFrameGenerator{
		width:  width,
		height: height,
		seed:   42, // Deterministic seed for reproducibility.
	}
}

// GenerateStaticFrame creates a static background frame for baseline testing.
//
// Arguments:
// - None.
//
// Returns:
// - A grayscale Mat containing a static pattern.
//
// @example
// frame := gen.GenerateStaticFrame()
// defer frame.Close()
func (g *MockFrameGenerator) GenerateStaticFrame() gocv.Mat {
	frame := gocv.NewMatWithSize(g.height, g.width, gocv.MatTypeCV8UC1)
	frame.SetTo(gocv.NewScalar(128, 0, 0, 0)) // Mid-gray background.
	return frame
}

// GenerateMotionFrame creates a frame with simulated motion at a specific position.
//
// Arguments:
// - x: X coordinate of motion region.
// - y: Y coordinate of motion region.
// - size: Size of the motion region in pixels.
//
// Returns:
// - A grayscale Mat containing simulated motion.
//
// @example
// frame := gen.GenerateMotionFrame(100, 100, 50)
// defer frame.Close()
func (g *MockFrameGenerator) GenerateMotionFrame(x, y, size int) gocv.Mat {
	frame := g.GenerateStaticFrame()

	// Add a bright region to simulate motion.
	rect := image.Rect(x, y, x+size, y+size)
	gocv.Rectangle(&frame, rect, color.RGBA{255, 255, 255, 0}, -1)

	return frame
}

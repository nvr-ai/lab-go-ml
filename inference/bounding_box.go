package inference

import (
	"fmt"
	"image"
)

// BoundingBox represents a bounding box with its label, confidence, and coordinates.
type BoundingBox struct {
	Label          string
	Confidence     float32
	X1, Y1, X2, Y2 float32
}

// This won't be entirely precise due to conversion to the integral rectangles
// from the image.Image library, but we're only using it to estimate which
// boxes are overlapping too much, so some imprecision should be OK.
func (b *BoundingBox) IOU(other *BoundingBox) float32 {
	return b.Intersection(other) / b.Union(other)
}

func (b *BoundingBox) String() string {
	return fmt.Sprintf("Object %s (confidence %f): (%f, %f), (%f, %f)",
		b.Label, b.Confidence, b.X1, b.Y1, b.X2, b.Y2)
}

// This loses precision, but recall that the boundingBox has already been
// scaled up to the original image's dimensions. So, it will only lose
// fractional pixels around the edges.
func (b *BoundingBox) ToRect() image.Rectangle {
	return image.Rect(int(b.X1), int(b.Y1), int(b.X2), int(b.Y2)).Canon()
}

// Returns the area of b in pixels, after converting to an image.Rectangle.
func (b *BoundingBox) RectArea() int {
	size := b.ToRect().Size()
	return size.X * size.Y
}

func (b *BoundingBox) Intersection(other *BoundingBox) float32 {
	r1 := b.ToRect()
	r2 := other.ToRect()
	intersected := r1.Intersect(r2).Canon().Size()
	return float32(intersected.X * intersected.Y)
}

func (b *BoundingBox) Union(other *BoundingBox) float32 {
	intersectArea := b.Intersection(other)
	totalArea := float32(b.RectArea() + other.RectArea())
	return totalArea - intersectArea
}

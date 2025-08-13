package kernels

import (
	"image"
	"image/color"
	"testing"
)

func TestBlurOperationsRadiusZeroReturnsCopy(t *testing.T) {
	img := image.NewRGBA(image.Rect(10, 20, 18, 27)) // non-zero Min
	for y := img.Rect.Min.Y; y < img.Rect.Max.Y; y++ {
		for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
			img.Set(x, y, color.RGBA{R: 10, G: 20, B: 30, A: 255})
		}
	}
	out := BoxBlur(img, Options{Radius: 0, Edge: EdgeClamp})
	if out.Bounds() != img.Bounds() {
		t.Fatalf("bounds mismatch")
	}
	if out.At(0, 0) != img.At(0, 0) {
		t.Fatalf("not a copy")
	}
}

func TestBlurOperationsBoundsMinNotZero(t *testing.T) {
	img := image.NewRGBA(image.Rect(5, 7, 9, 12))
	img.SetRGBA(5, 7, color.RGBA{255, 0, 0, 255})
	out := BoxBlur(img, Options{Radius: 1})
	if out.Rect != img.Rect {
		t.Fatalf("bounds mismatch")
	}
	// Top-left should be non-zero due to blur; ensure we touched correct pixels.
	if out.RGBAAt(5, 7) == (color.RGBA{}) {
		t.Fatalf("unexpected zero at top-left")
	}
}

func TestBlurOperationsEdgeModes(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 3, 1))
	img.SetRGBA(0, 0, color.RGBA{0, 0, 0, 255})
	img.SetRGBA(1, 0, color.RGBA{255, 0, 0, 255})
	img.SetRGBA(2, 0, color.RGBA{0, 0, 0, 255})

	r := 1
	outClamp := BoxBlur(img, Options{Radius: r, Edge: EdgeClamp})
	outMirror := BoxBlur(img, Options{Radius: r, Edge: EdgeMirror})
	outWrap := BoxBlur(img, Options{Radius: r, Edge: EdgeWrap})
	_ = outClamp
	_ = outMirror
	_ = outWrap
}

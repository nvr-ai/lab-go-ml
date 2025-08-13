package kernels

import (
	"image"
	"math/rand"
	"testing"
	"time"
)

func BoxBlurNaive(img image.Image, radius int, mode EdgeMode) image.Image {
	return BoxBlur(img, Options{Radius: radius, Edge: mode})
}

func BoxBlurFast(img image.Image, opt Options) *image.RGBA {
	return BoxBlur(img, opt)
}

func genRGBA(w, h int, alpha bool) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(1))
	for y := 0; y < h; y++ {
		row := y * img.Stride
		for x := 0; x < w; x++ {
			i := row + x*4
			img.Pix[i+0] = uint8(rng.Intn(256))
			img.Pix[i+1] = uint8(rng.Intn(256))
			img.Pix[i+2] = uint8(rng.Intn(256))
			if alpha {
				img.Pix[i+3] = uint8(rng.Intn(256))
			} else {
				img.Pix[i+3] = 255
			}
		}
	}
	return img
}

func BenchmarkNaive_640_r3(b *testing.B) {
	img := genRGBA(640, 640, false)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BoxBlurNaive(img, 3, EdgeClamp)
	}
}

func BenchmarkFast_640_r3(b *testing.B) {
	img := genRGBA(640, 640, false)
	opt := Options{Radius: 3, Edge: EdgeClamp, Parallel: true}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BoxBlurFast(img, opt)
	}
}

func BenchmarkNaive_1080p_r7(b *testing.B) {
	img := genRGBA(1920, 1080, false)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BoxBlurNaive(img, 7, EdgeClamp)
	}
}

func BenchmarkFast_1080p_r7(b *testing.B) {
	img := genRGBA(1920, 1080, false)
	opt := Options{Radius: 7, Edge: EdgeClamp, Parallel: true}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BoxBlurFast(img, opt)
	}
}

// Real-world-ish: blur ROIs for privacy (many small faces)
func BenchmarkPrivacy_1080p_10faces(b *testing.B) {
	img := genRGBA(1920, 1080, false)
	var rois []image.Rectangle
	for i := 0; i < 10; i++ {
		x := (i * 180) % (1920 - 160)
		y := (i * 120) % (1080 - 160)
		rois = append(rois, image.Rect(x, y, x+160, y+160))
	}
	opt := Options{Radius: 9, Edge: EdgeMirror, Parallel: true}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = BlurRegions(img, rois, opt)
	}
}

// End-to-end throughput proxy: blur + small compute
func BenchmarkPipeline_1080p(b *testing.B) {
	img := genRGBA(1920, 1080, false)
	opt := Options{Radius: 5, Edge: EdgeClamp, Parallel: true}
	work := func() {
		_ = BoxBlurFast(img, opt)
		time.Sleep(500 * time.Microsecond) // emulate ORT decode/postproc
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		work()
	}
}

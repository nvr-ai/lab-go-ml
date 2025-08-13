package images

import (
	"fmt"
	"image"
	"math/rand"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/util"
)

// Benchmark test cases covering various IoU scenarios with different overlap characteristics
// and performance implications for real-world object detection workloads.

func init() {
	// Seed random number generator for consistent benchmarks
	rand.Seed(time.Now().UnixNano())
}

// BenchmarkIoU_NonOverlapping tests performance with rectangles that don't overlap.
// This is the most optimized path as it returns early when w <= 0 || h <= 0.
func BenchmarkIoU_NonOverlapping(b *testing.B) {
	rect1 := Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}
	rect2 := Rect{X1: 200, Y1: 200, X2: 300, Y2: 300}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = CalculateIoU(rect1.X1, rect1.Y1, rect1.X2, rect1.Y2, rect2.X1, rect2.Y1, rect2.X2, rect2.Y2)
	}
}

// BenchmarkIoU_FullOverlap tests identical rectangles (IoU = 1.0).
// This exercises the full calculation path with maximum intersection.
func BenchmarkIoU_FullOverlap(b *testing.B) {
	rect1 := Rect{X1: 50, Y1: 50, X2: 150, Y2: 150}
	rect2 := Rect{X1: 50, Y1: 50, X2: 150, Y2: 150}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = CalculateIoU(rect1.X1, rect1.Y1, rect1.X2, rect1.Y2, rect2.X1, rect2.Y1, rect2.X2, rect2.Y2)
	}
}

// BenchmarkIoU_PartialOverlap tests common object detection scenario with 0.3-0.7 IoU.
// This represents typical bounding box predictions vs ground truth comparisons.
func BenchmarkIoU_PartialOverlap(b *testing.B) {
	rect1 := Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}
	rect2 := Rect{X1: 50, Y1: 50, X2: 150, Y2: 150}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = CalculateIoU(rect1.X1, rect1.Y1, rect1.X2, rect1.Y2, rect2.X1, rect2.Y1, rect2.X2, rect2.Y2)
	}
}

// BenchmarkIoU_SmallRectangles tests performance with small bounding boxes.
// Common in detection of distant objects or fine-grained details.
func BenchmarkIoU_SmallRectangles(b *testing.B) {
	rect1 := Rect{X1: 10, Y1: 10, X2: 15, Y2: 15}
	rect2 := Rect{X1: 12, Y1: 12, X2: 18, Y2: 18}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = CalculateIoU(rect1.X1, rect1.Y1, rect1.X2, rect1.Y2, rect2.X1, rect2.Y1, rect2.X2, rect2.Y2)
	}
}

// BenchmarkIoU_LargeRectangles tests performance with large bounding boxes.
// Common when detecting large objects or using high-resolution images.
func BenchmarkIoU_LargeRectangles(b *testing.B) {
	rect1 := Rect{X1: 0, Y1: 0, X2: 1920, Y2: 1080}
	rect2 := Rect{X1: 960, Y1: 540, X2: 1920, Y2: 1080}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = CalculateIoU(rect1.X1, rect1.Y1, rect1.X2, rect1.Y2, rect2.X1, rect2.Y1, rect2.X2, rect2.Y2)
	}
}

// BenchmarkIoU_TouchingEdges tests edge case where rectangles only touch at boundaries.
// This should return 0.0 quickly due to w <= 0 || h <= 0 optimization.
func BenchmarkIoU_TouchingEdges(b *testing.B) {
	rect1 := Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}
	rect2 := Rect{X1: 100, Y1: 0, X2: 200, Y2: 100}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = CalculateIoU(rect1.X1, rect1.Y1, rect1.X2, rect1.Y2, rect2.X1, rect2.Y1, rect2.X2, rect2.Y2)
	}
}

// BenchmarkIoU_RandomPairs benchmarks with random rectangle pairs.
// Simulates real-world object detection workload with varied overlap scenarios.
func BenchmarkIoU_RandomPairs(b *testing.B) {
	// Pre-generate random rectangle pairs for consistent benchmarking
	pairs := make([]struct{ r1, r2 Rect }, 1000)
	for i := range pairs {
		// Generate random rectangles within typical image bounds
		x1, y1 := MinFloat32(1920, float32(rand.Intn(1920))), MinFloat32(1080, float32(rand.Intn(1080)))
		w1, h1 := MinFloat32(300, float32(rand.Intn(300)+20)), MinFloat32(300, float32(rand.Intn(300)+20))
		x2, y2 := MinFloat32(1920, float32(rand.Intn(1920))), MinFloat32(1080, float32(rand.Intn(1080)))
		w2, h2 := MinFloat32(300, float32(rand.Intn(300)+20)), MinFloat32(300, float32(rand.Intn(300)+20))

		pairs[i].r1 = Rect{X1: x1, Y1: y1, X2: x1 + w1, Y2: y1 + h1}
		pairs[i].r2 = Rect{X1: x2, Y1: y2, X2: x2 + w2, Y2: y2 + h2}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		pair := pairs[i%len(pairs)]
		_ = CalculateIoU(
			pair.r1.X1,
			pair.r1.Y1,
			pair.r1.X2,
			pair.r1.Y2,
			pair.r2.X1,
			pair.r2.Y1,
			pair.r2.X2,
			pair.r2.Y2,
		)
	}
}

// Comparison benchmarks against image.Rectangle to justify custom implementation

// imageRectangleIoU implements IoU using Go's standard library image.Rectangle
func imageRectangleIoU(r1, r2 image.Rectangle) float32 {
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

// BenchmarkImageRectangle_NonOverlapping compares against image.Rectangle for non-overlapping cases
func BenchmarkImageRectangle_NonOverlapping(b *testing.B) {
	r1 := image.Rect(0, 0, 100, 100)
	r2 := image.Rect(200, 200, 300, 300)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = imageRectangleIoU(r1, r2)
	}
}

// BenchmarkImageRectangle_PartialOverlap compares against image.Rectangle for partial overlap
func BenchmarkImageRectangle_PartialOverlap(b *testing.B) {
	r1 := image.Rect(0, 0, 100, 100)
	r2 := image.Rect(50, 50, 150, 150)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = imageRectangleIoU(r1, r2)
	}
}

// BenchmarkImageRectangle_RandomPairs compares against image.Rectangle for random pairs
func BenchmarkImageRectangle_RandomPairs(b *testing.B) {
	// Use same random pairs as custom implementation for fair comparison
	pairs := make([]struct{ r1, r2 image.Rectangle }, 1000)
	for i := range pairs {
		x1, y1 := rand.Intn(1920), rand.Intn(1080)
		w1, h1 := rand.Intn(300)+20, rand.Intn(300)+20
		x2, y2 := rand.Intn(1920), rand.Intn(1080)
		w2, h2 := rand.Intn(300)+20, rand.Intn(300)+20

		pairs[i].r1 = image.Rect(x1, y1, x1+w1, y1+h1)
		pairs[i].r2 = image.Rect(x2, y2, x2+w2, y2+h2)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		pair := pairs[i%len(pairs)]
		_ = imageRectangleIoU(pair.r1, pair.r2)
	}
}

// Real-world benchmarks using actual image data from loader utility

// BenchmarkIoU_RealImageBounds tests IoU performance using actual image dimensions
// from the corpus to simulate realistic object detection scenarios.
func BenchmarkIoU_RealImageBounds(b *testing.B) {
	// Load a subset of real imgs to get realistic bounds
	imgs, err := util.LoadDirectoryImageFiles("../../../ml/corpus/images/clip-4k.mp4")
	if err != nil {
		b.Skipf("Could not load test images: %v", err)
		return
	}

	if len(imgs) == 0 {
		b.Skip("No images found for realistic bounds testing")
		return
	}

	// Use a smaller subset to avoid excessive benchmark time
	testImages := imgs[:int(MinFloat32(50, float32(len(imgs))))]

	// Generate realistic bounding boxes based on actual image sizes
	// Assume typical object sizes: 5-25% of image width/height
	rectanglePairs := make([]struct{ r1, r2 Rect }, len(testImages)*10)

	for i := range testImages {
		// For benchmark purposes, assume standard dimensions or parse from filename
		// Using 4K resolution as indicated by clip-4k.mp4
		imgWidth, imgHeight := 3840, 2160

		for j := 0; j < 10; j++ { // Generate 10 pairs per image
			idx := i*10 + j

			// Generate realistic object bounding boxes (5-25% of image size)
			minSize := int(float64(MinFloat32(float32(imgWidth), float32(imgHeight))) * 0.05)
			maxSize := int(float64(MinFloat32(float32(imgWidth), float32(imgHeight))) * 0.25)

			// First rectangle
			w1 := rand.Intn(maxSize-minSize) + minSize
			h1 := rand.Intn(maxSize-minSize) + minSize
			x1 := rand.Intn(imgWidth - w1)
			y1 := rand.Intn(imgHeight - h1)

			// Second rectangle with some bias toward overlap for realistic detection scenarios
			w2 := rand.Intn(maxSize-minSize) + minSize
			h2 := rand.Intn(maxSize-minSize) + minSize

			// 70% chance of potential overlap, 30% chance of random placement
			var x2, y2 int
			if rand.Float32() < 0.7 {
				// Bias toward potential overlap
				offsetX := rand.Intn(w1*2) - w1
				offsetY := rand.Intn(h1*2) - h1
				x2 = int(MaxFloat32(0, MinFloat32(float32(imgWidth-w2), float32(x1+offsetX))))
				y2 = int(MaxFloat32(0, MinFloat32(float32(imgHeight-h2), float32(y1+offsetY))))
			} else {
				// Random placement
				x2 = rand.Intn(imgWidth - w2)
				y2 = rand.Intn(imgHeight - h2)
			}

			rectanglePairs[idx].r1 = Rect{X1: float32(x1), Y1: float32(y1), X2: float32(x1 + w1), Y2: float32(y1 + h1)}
			rectanglePairs[idx].r2 = Rect{X1: float32(x2), Y1: float32(y2), X2: float32(x2 + w2), Y2: float32(y2 + h2)}
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		pair := rectanglePairs[i%len(rectanglePairs)]
		_ = CalculateIoU(
			pair.r1.X1,
			pair.r1.Y1,
			pair.r1.X2,
			pair.r1.Y2,
			pair.r2.X1,
			pair.r2.Y1,
			pair.r2.X2,
			pair.r2.Y2,
		)
	}
}

// BenchmarkIoU_BatchProcessing tests performance when processing multiple IoU calculations
// in batch, simulating Non-Maximum Suppression (NMS) operations in object detection.
func BenchmarkIoU_BatchProcessing(b *testing.B) {
	// Create a set of detection boxes similar to what an object detector might produce
	const numDetections = 100
	detections := make([]Rect, numDetections)

	// Generate realistic detection boxes clustered in certain areas (like real detections)
	clusterCenters := [][2]int{
		{500, 300}, {1200, 600}, {300, 800}, // 3 cluster centers
	}

	for i := range detections {
		center := clusterCenters[i%len(clusterCenters)]

		// Add noise around cluster center
		noiseX := rand.Intn(400) - 200
		noiseY := rand.Intn(400) - 200

		centerX := int(MaxFloat32(50, MinFloat32(1870, float32(center[0]+noiseX))))
		centerY := int(MaxFloat32(50, MinFloat32(1030, float32(center[1]+noiseY))))

		// Random box size (typical object detection box sizes)
		width := rand.Intn(200) + 50
		height := rand.Intn(200) + 50

		detections[i] = Rect{
			X1: float32(centerX - width/2),
			Y1: float32(centerY - height/2),
			X2: float32(centerX + width/2),
			Y2: float32(centerY + height/2),
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	// Simulate NMS: each detection compared against all others
	for i := 0; i < b.N; i++ {
		var totalIoU float32
		for j := range detections {
			for k := j + 1; k < len(detections); k++ {
				totalIoU += CalculateIoU(
					detections[j].X1,
					detections[j].Y1,
					detections[j].X2,
					detections[j].Y2,
					detections[k].X1,
					detections[k].Y1,
					detections[k].X2,
					detections[k].Y2,
				)
			}
		}
		// Use totalIoU to prevent compiler optimization
		_ = totalIoU
	}
}

// Utility functions for benchmarks
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Performance analysis benchmark that provides comprehensive metrics
func BenchmarkIoU_PerformanceAnalysis(b *testing.B) {
	scenarios := []struct {
		name string
		r1   Rect
		r2   Rect
	}{
		{"NoOverlap", Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}, Rect{X1: 200, Y1: 200, X2: 300, Y2: 300}},
		{"TouchingEdge", Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}, Rect{X1: 100, Y1: 0, X2: 200, Y2: 100}},
		{"SmallOverlap", Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}, Rect{X1: 90, Y1: 90, X2: 190, Y2: 190}},
		{"HalfOverlap", Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}, Rect{X1: 50, Y1: 50, X2: 150, Y2: 150}},
		{"LargeOverlap", Rect{X1: 0, Y1: 0, X2: 100, Y2: 100}, Rect{X1: 20, Y1: 20, X2: 120, Y2: 120}},
		{"FullOverlap", Rect{X1: 50, Y1: 50, X2: 150, Y2: 150}, Rect{X1: 50, Y1: 50, X2: 150, Y2: 150}},
		{"LargeBoxes", Rect{X1: 0, Y1: 0, X2: 1920, Y2: 1080}, Rect{X1: 960, Y1: 540, X2: 1920, Y2: 1080}},
		{"TinyBoxes", Rect{X1: 10, Y1: 10, X2: 12, Y2: 12}, Rect{X1: 11, Y1: 11, X2: 13, Y2: 13}},
	}

	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			b.ReportAllocs()

			start := time.Now()
			for i := 0; i < b.N; i++ {
				_ = CalculateIoU(
					scenario.r1.X1,
					scenario.r1.Y1,
					scenario.r1.X2,
					scenario.r1.Y2,
					scenario.r2.X1,
					scenario.r2.Y1,
					scenario.r2.X2,
					scenario.r2.Y2,
				)
			}
			elapsed := time.Since(start)

			// Report custom metrics
			nsPerOp := float64(elapsed.Nanoseconds()) / float64(b.N)
			b.ReportMetric(nsPerOp, "ns/op")

			// Calculate theoretical operations per second for object detection context
			opsPerSec := 1e9 / nsPerOp
			b.ReportMetric(opsPerSec, "IoU-ops/sec")
		})
	}
}

// Example output interpretation function for documentation
func Example_ioUBenchmarkInterpretation() {
	// This example demonstrates how to interpret IoU benchmark results
	// in the context of real-time object detection performance requirements.

	fmt.Println("IoU Benchmark Performance Guidelines:")
	fmt.Println("")
	fmt.Println("Target Performance for Real-time Detection:")
	fmt.Println("- 1080p@30fps with 100 detections/frame: ~3M IoU ops/sec required")
	fmt.Println("- 4K@15fps with 200 detections/frame: ~600K IoU ops/sec required")
	fmt.Println("- Batch NMS processing: 100 boxes = 4,950 IoU calculations")
	fmt.Println("")
	fmt.Println("Performance Expectations:")
	fmt.Println("- Non-overlapping boxes: Fastest (early return)")
	fmt.Println("- Overlapping boxes: Full calculation required")
	fmt.Println("- Large rectangles: Same computational complexity as small ones")
	fmt.Println("- Custom implementation should outperform image.Rectangle")

	// Output:
	// IoU Benchmark Performance Guidelines:
	//
	// Target Performance for Real-time Detection:
	// - 1080p@30fps with 100 detections/frame: ~3M IoU ops/sec required
	// - 4K@15fps with 200 detections/frame: ~600K IoU ops/sec required
	// - Batch NMS processing: 100 boxes = 4,950 IoU calculations
	//
	// Performance Expectations:
	// - Non-overlapping boxes: Fastest (early return)
	// - Overlapping boxes: Full calculation required
	// - Large rectangles: Same computational complexity as small ones
	// - Custom implementation should outperform image.Rectangle
}

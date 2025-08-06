package dfine

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"runtime"
	"sort"

	"github.com/8ff/prettyTimer"
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	modelPath = "./dfine_dynamic.onnx"
	imagePath = "../../../../../ml/corpus/frame-75-480p.jpg"
	useCoreML = false
)

// DFINEModelSession represents a D-FINE object detection session with dynamic input support.
//
// This structure manages the ONNX Runtime session and tensors for the D-FINE model,
// supporting variable input dimensions for different image sizes.
type DFINEModelSession struct {
	Session      *ort.AdvancedSession
	FeatureMaps  []*ort.Tensor[float32] // Multiple feature maps at different scales
	OutputLogits *ort.Tensor[float32]   // Classification scores
	OutputBoxes  *ort.Tensor[float32]   // Bounding box predictions
	FeatStrides  []int                  // Feature strides for each level
	FeatChannels []int                  // Feature channels for each level
	NumQueries   int                    // Number of object queries
	NumClasses   int                    // Number of classes
}

// BoundingBox represents a detected object with its location and confidence.
//
// This structure contains all information about a detected object including
// its class label, confidence score, and bounding box coordinates.
type BoundingBox struct {
	label      string
	confidence float32
	x1, y1     float32
	x2, y2     float32
}

// String formats the bounding box information for display.
//
// Returns:
// - A formatted string containing object class, confidence, and coordinates.
//
// @example
// box := BoundingBox{label: "person", confidence: 0.95, x1: 100, y1: 100, x2: 200, y2: 300}
// fmt.Println(box.String()) // Output: Object person (confidence 0.950000): (100.00, 100.00), (200.00, 300.00)
func (b *BoundingBox) String() string {
	return fmt.Sprintf("Object %s (confidence %f): (%.2f, %.2f), (%.2f, %.2f)",
		b.label, b.confidence, b.x1, b.y1, b.x2, b.y2)
}

// ToRect converts the bounding box to an image.Rectangle.
//
// This method converts floating-point coordinates to integer coordinates
// suitable for image processing operations.
//
// Returns:
// - An image.Rectangle with canonicalized coordinates.
//
// @example
// box := BoundingBox{x1: 100.5, y1: 100.5, x2: 200.5, y2: 300.5}
// rect := box.ToRect()
// fmt.Printf("Rectangle: %v\n", rect) // Rectangle: (100,100)-(201,301)
func (b *BoundingBox) ToRect() image.Rectangle {
	return image.Rect(int(b.x1), int(b.y1), int(b.x2), int(b.y2)).Canon()
}

// Intersection calculates the intersection area between two bounding boxes.
//
// Arguments:
// - other: The other bounding box to calculate intersection with.
//
// Returns:
// - The area of intersection in pixels as float32.
//
// @example
// box1 := BoundingBox{x1: 0, y1: 0, x2: 100, y2: 100}
// box2 := BoundingBox{x1: 50, y1: 50, x2: 150, y2: 150}
// area := box1.Intersection(&box2) // Returns 2500.0 (50x50 overlap)
func (b *BoundingBox) Intersection(other *BoundingBox) float32 {
	r1 := b.ToRect()
	r2 := other.ToRect()
	intersected := r1.Intersect(r2).Canon().Size()
	return float32(intersected.X * intersected.Y)
}

// Union calculates the union area between two bounding boxes.
//
// Arguments:
// - other: The other bounding box to calculate union with.
//
// Returns:
// - The area of union in pixels as float32.
//
// @example
// box1 := BoundingBox{x1: 0, y1: 0, x2: 100, y2: 100}
// box2 := BoundingBox{x1: 50, y1: 50, x2: 150, y2: 150}
// area := box1.Union(&box2) // Returns 17500.0
func (b *BoundingBox) Union(other *BoundingBox) float32 {
	intersectArea := b.Intersection(other)
	r1 := b.ToRect()
	r2 := other.ToRect()
	size1 := r1.Size()
	size2 := r2.Size()
	totalArea := float32(size1.X*size1.Y + size2.X*size2.Y)
	return totalArea - intersectArea
}

// IoU calculates the Intersection over Union between two bounding boxes.
//
// This metric is used for Non-Maximum Suppression (NMS) to remove duplicate detections.
//
// Arguments:
// - other: The other bounding box to calculate IoU with.
//
// Returns:
// - The IoU value between 0 and 1.
//
// @example
// box1 := BoundingBox{x1: 0, y1: 0, x2: 100, y2: 100}
// box2 := BoundingBox{x1: 50, y1: 50, x2: 150, y2: 150}
// iou := box1.IoU(&box2) // Returns ~0.143 (2500/17500)
func (b *BoundingBox) IoU(other *BoundingBox) float32 {
	return b.Intersection(other) / b.Union(other)
}

// ExtractMultiScaleFeatures extracts features at multiple scales from an input image.
//
// This function simulates a Feature Pyramid Network (FPN) backbone by creating
// downsampled feature maps at different scales. In a real implementation, these
// would come from a CNN backbone like ResNet.
//
// Arguments:
// - img: The input image to extract features from.
// - session: The D-FINE model session containing feature specifications.
//
// Returns:
// - An error if feature extraction fails, nil otherwise.
//
// @example
// pic, _ := loadImageFile("image.jpg")
// session := &DFINEModelSession{FeatStrides: []int{8, 16, 32}, ...}
// err := ExtractMultiScaleFeatures(pic, session)
func ExtractMultiScaleFeatures(img image.Image, session *DFINEModelSession) error {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()

	// For each feature level, create appropriate feature maps
	for i, stride := range session.FeatStrides {
		featWidth := width / stride
		featHeight := height / stride
		channels := session.FeatChannels[i]

		// Get tensor data slice
		data := session.FeatureMaps[i].GetData()
		expectedSize := channels * featHeight * featWidth
		if len(data) < expectedSize {
			return fmt.Errorf("Feature tensor %d too small: has %d, needs %d",
				i, len(data), expectedSize)
		}

		// In a real implementation, this would be CNN features
		// For now, we'll create a simplified representation
		resized := resize.Resize(uint(featWidth), uint(featHeight), img, resize.Lanczos3)

		// Fill the tensor with normalized pixel values
		// This is a placeholder - real features would come from a backbone network
		idx := 0
		for c := 0; c < channels; c++ {
			for y := 0; y < featHeight; y++ {
				for x := 0; x < featWidth; x++ {
					if c < 3 { // Use RGB channels if available
						r, g, b, _ := resized.At(x, y).RGBA()
						switch c {
						case 0:
							data[idx] = float32(r>>8) / 255.0
						case 1:
							data[idx] = float32(g>>8) / 255.0
						case 2:
							data[idx] = float32(b>>8) / 255.0
						}
					} else {
						// Fill other channels with learned features (placeholder)
						data[idx] = 0.0
					}
					idx++
				}
			}
		}
	}

	return nil
}

// processDFINEOutput processes the raw outputs from D-FINE model.
//
// This function converts the model's predictions into bounding boxes,
// applies confidence thresholding, and performs Non-Maximum Suppression.
//
// Arguments:
// - logits: Raw classification scores from the model [batch, num_queries, num_classes].
// - boxes: Raw bounding box predictions [batch, num_queries, 4].
// - numClasses: Number of object classes.
// - originalWidth: Original image width for coordinate scaling.
// - originalHeight: Original image height for coordinate scaling.
// - confThreshold: Confidence threshold for filtering detections.
// - nmsThreshold: IoU threshold for Non-Maximum Suppression.
//
// Returns:
// - A slice of BoundingBox objects representing detected objects.
//
// @example
// logits := outputLogits.GetData()
// boxes := outputBoxes.GetData()
// detections := processDFINEOutput(logits, boxes, 80, 640, 480, 0.5, 0.7)
func ProcessDFINEOutput(
	logits []float32,
	boxes []float32,
	numClasses int,
	originalWidth int,
	originalHeight int,
	confThreshold float32,
	nmsThreshold float32,
) []BoundingBox {
	numQueries := len(logits) / numClasses
	boundingBoxes := make([]BoundingBox, 0, numQueries)

	// Process each query
	for q := 0; q < numQueries; q++ {
		// Find the class with highest probability
		maxProb := float32(-1e9)
		classID := -1

		for c := 0; c < numClasses; c++ {
			prob := logits[q*numClasses+c]
			if prob > maxProb {
				maxProb = prob
				classID = c
			}
		}

		// Apply sigmoid to get probability
		confidence := float32(1.0 / (1.0 + math.Exp(float64(-maxProb))))

		// Skip if below threshold
		if confidence < confThreshold {
			continue
		}

		// Extract bounding box (cx, cy, w, h format)
		boxIdx := q * 4
		cx := boxes[boxIdx] * float32(originalWidth)
		cy := boxes[boxIdx+1] * float32(originalHeight)
		w := boxes[boxIdx+2] * float32(originalWidth)
		h := boxes[boxIdx+3] * float32(originalHeight)

		// Convert to x1, y1, x2, y2 format
		x1 := cx - w/2
		y1 := cy - h/2
		x2 := cx + w/2
		y2 := cy + h/2

		// Clamp to image boundaries
		x1 = max(0, min(x1, float32(originalWidth)))
		y1 = max(0, min(y1, float32(originalHeight)))
		x2 = max(0, min(x2, float32(originalWidth)))
		y2 = max(0, min(y2, float32(originalHeight)))

		boundingBoxes = append(boundingBoxes, BoundingBox{
			label:      cocoClasses[classID],
			confidence: confidence,
			x1:         x1,
			y1:         y1,
			x2:         x2,
			y2:         y2,
		})
	}

	// Sort by confidence (descending)
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence > boundingBoxes[j].confidence
	})

	// Apply Non-Maximum Suppression
	mergedResults := make([]BoundingBox, 0, len(boundingBoxes))
	for _, candidateBox := range boundingBoxes {
		overlapsExistingBox := false
		for _, existingBox := range mergedResults {
			if candidateBox.IoU(&existingBox) > nmsThreshold {
				overlapsExistingBox = true
				break
			}
		}
		if !overlapsExistingBox {
			mergedResults = append(mergedResults, candidateBox)
		}
	}

	return mergedResults
}

// InitDFINESession initializes a D-FINE detection session with dynamic input support.
//
// This function creates an ONNX Runtime session for the D-FINE model,
// supporting variable input dimensions for different image sizes.
//
// Arguments:
// - modelPath: Path to the D-FINE ONNX model file.
// - width: Input image width.
// - height: Input image height.
// - featStrides: Feature strides for each pyramid level (e.g., []int{8, 16, 32}).
// - featChannels: Feature channels for each level (e.g., []int{512, 1024, 2048}).
//
// Returns:
// - A pointer to the initialized DFINEModelSession.
// - An error if initialization fails.
//
// @example
// session, err := InitDFINESession("dfine.onnx", 640, 640, []int{8, 16, 32}, []int{512, 1024, 2048})
//
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// defer session.Destroy()
func InitDFINESession(
	modelPath string,
	width, height int,
	featStrides []int,
	featChannels []int,
) (*DFINEModelSession, error) {
	ort.SetSharedLibraryPath(getSharedLibPath())
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("Error initializing ORT environment: %w", err)
	}

	// Create feature map tensors for each level
	featureMaps := make([]*ort.Tensor[float32], len(featStrides))
	inputNames := make([]string, len(featStrides))

	for i, stride := range featStrides {
		featHeight := height / stride
		featWidth := width / stride
		channels := featChannels[i]

		shape := ort.NewShape(1, int64(channels), int64(featHeight), int64(featWidth))
		tensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			// Clean up previously created tensors
			for j := 0; j < i; j++ {
				featureMaps[j].Destroy()
			}
			return nil, fmt.Errorf("Error creating feature tensor %d: %w", i, err)
		}
		featureMaps[i] = tensor
		inputNames[i] = fmt.Sprintf("feat%d", i)
	}

	// Create output tensors
	// Assuming 300 queries and 80 classes (COCO dataset)
	numQueries := 300
	numClasses := 80

	logitsShape := ort.NewShape(1, int64(numQueries), int64(numClasses))
	outputLogits, err := ort.NewEmptyTensor[float32](logitsShape)
	if err != nil {
		for _, t := range featureMaps {
			t.Destroy()
		}
		return nil, fmt.Errorf("Error creating logits tensor: %w", err)
	}

	boxesShape := ort.NewShape(1, int64(numQueries), 4)
	outputBoxes, err := ort.NewEmptyTensor[float32](boxesShape)
	if err != nil {
		for _, t := range featureMaps {
			t.Destroy()
		}
		outputLogits.Destroy()
		return nil, fmt.Errorf("Error creating boxes tensor: %w", err)
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		for _, t := range featureMaps {
			t.Destroy()
		}
		outputLogits.Destroy()
		outputBoxes.Destroy()
		return nil, fmt.Errorf("Error creating ORT session options: %w", err)
	}
	defer options.Destroy()

	// Enable CoreML if requested
	if useCoreML {
		err = options.AppendExecutionProviderCoreML(0)
		if err != nil {
			for _, t := range featureMaps {
				t.Destroy()
			}
			outputLogits.Destroy()
			outputBoxes.Destroy()
			return nil, fmt.Errorf("Error enabling CoreML: %w", err)
		}
	}

	// Create advanced session
	inputTensors := make([]ort.ArbitraryTensor, len(featureMaps))
	for i, t := range featureMaps {
		inputTensors[i] = t
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		[]string{"pred_logits", "pred_boxes"},
		inputTensors,
		[]ort.ArbitraryTensor{outputLogits, outputBoxes},
		options,
	)
	if err != nil {
		for _, t := range featureMaps {
			t.Destroy()
		}
		outputLogits.Destroy()
		outputBoxes.Destroy()
		return nil, fmt.Errorf("Error creating ORT session: %w", err)
	}

	return &DFINEModelSession{
		Session:      session,
		FeatureMaps:  featureMaps,
		OutputLogits: outputLogits,
		OutputBoxes:  outputBoxes,
		FeatStrides:  featStrides,
		FeatChannels: featChannels,
		NumQueries:   numQueries,
		NumClasses:   numClasses,
	}, nil
}

// Destroy releases all resources associated with the D-FINE session.
//
// This method must be called when the session is no longer needed
// to prevent memory leaks.
//
// @example
// session, _ := initDFINESession(...)
// defer session.Destroy()
func (m *DFINEModelSession) Destroy() {
	m.Session.Destroy()
	for _, feat := range m.FeatureMaps {
		feat.Destroy()
	}
	m.OutputLogits.Destroy()
	m.OutputBoxes.Destroy()
}

// loadImageFile loads an image from disk.
//
// Arguments:
// - filePath: Path to the image file.
//
// Returns:
// - The loaded image as an image.Image interface.
// - An error if loading fails.
//
// @example
// img, err := loadImageFile("test.jpg")
//
//	if err != nil {
//	    log.Fatal(err)
//	}
func loadImageFile(filePath string) (image.Image, error) {
	f, e := os.Open(filePath)
	if e != nil {
		return nil, fmt.Errorf("Error opening %s: %w", filePath, e)
	}
	defer f.Close()
	pic, _, e := image.Decode(f)
	if e != nil {
		return nil, fmt.Errorf("Error decoding %s: %w", filePath, e)
	}
	return pic, nil
}

// getSharedLibPath returns the appropriate ONNX Runtime library path for the current platform.
//
// Returns:
// - The file path to the ONNX Runtime shared library.
//
// @example
// libPath := getSharedLibPath()
// fmt.Printf("Using ONNX Runtime library: %s\n", libPath)
func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime_amd64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so"
		}
		return "../third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

// min returns the minimum of two float32 values.
//
// Arguments:
// - a: First value.
// - b: Second value.
//
// Returns:
// - The smaller of the two values.
//
// @example
// result := min(3.14, 2.71) // Returns 2.71
func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two float32 values.
//
// Arguments:
// - a: First value.
// - b: Second value.
//
// Returns:
// - The larger of the two values.
//
// @example
// result := max(3.14, 2.71) // Returns 3.14
func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Array of COCO dataset class labels
var cocoClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
	"giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
	"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
	"toothbrush",
}

func main() {
	os.Exit(Run())
}

// Run executes the main detection pipeline.
//
// Returns:
// - Exit code (0 for success, 1 for error).
//
// @example
// exitCode := Run()
// os.Exit(exitCode)
func Run() int {
	timingStats := prettyTimer.NewTimingStats()

	if os.Getenv("USE_COREML") == "true" {
		useCoreML = true
	}

	// Read the input image
	pic, e := loadImageFile(imagePath)
	if e != nil {
		fmt.Printf("Error loading input image: %s\n", e)
		return 1
	}
	originalWidth := pic.Bounds().Canon().Dx()
	originalHeight := pic.Bounds().Canon().Dy()

	fmt.Printf("Loaded image: %dx%d\n", originalWidth, originalHeight)

	// Initialize D-FINE session with typical feature pyramid settings
	// These should match the backbone used during model export
	featStrides := []int{8, 16, 32}
	featChannels := []int{512, 1024, 2048}

	modelSession, e := InitDFINESession(
		modelPath,
		originalWidth,
		originalHeight,
		featStrides,
		featChannels,
	)
	if e != nil {
		fmt.Printf("Error creating session: %s\n", e)
		return 1
	}
	defer modelSession.Destroy()

	// Run detection 5 times for timing statistics
	for i := 0; i < 5; i++ {
		// Extract features (in real usage, this would come from a backbone CNN)
		e = ExtractMultiScaleFeatures(pic, modelSession)
		if e != nil {
			fmt.Printf("Error extracting features: %s\n", e)
			return 1
		}

		timingStats.Start()
		e = modelSession.Session.Run()
		if e != nil {
			fmt.Printf("Error running ORT session: %s\n", e)
			return 1
		}
		timingStats.Finish()

		// Process outputs
		logits := modelSession.OutputLogits.GetData()
		boxes := modelSession.OutputBoxes.GetData()

		detections := ProcessDFINEOutput(
			logits,
			boxes,
			modelSession.NumClasses,
			originalWidth,
			originalHeight,
			0.5, // Confidence threshold
			0.7, // NMS threshold
		)

		// Print results
		fmt.Printf("\n--- Detection %d Results ---\n", i+1)
		for idx, box := range detections {
			fmt.Printf("Detection %d: %s\n", idx, &box)
		}
		fmt.Printf("Total detections: %d\n", len(detections))
	}

	timingStats.PrintStats()
	return 0
}

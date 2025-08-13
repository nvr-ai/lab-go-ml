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
	"github.com/nvr-ai/go-ml/common"
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

// ProcessDFINEOutput processes the raw outputs from D-FINE model.
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
) []common.BoundingBox {
	numQueries := len(logits) / numClasses
	boundingBoxes := make([]common.BoundingBox, 0, numQueries)

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

		boundingBoxes = append(boundingBoxes, common.BoundingBox{
			Label:      cocoClasses[classID],
			Confidence: confidence,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		})
	}

	// Sort by confidence (descending)
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].Confidence > boundingBoxes[j].Confidence
	})

	// Apply Non-Maximum Suppression
	mergedResults := make([]common.BoundingBox, 0, len(boundingBoxes))
	for _, candidateBox := range boundingBoxes {
		overlapsExistingBox := false
		for _, existingBox := range mergedResults {
			if candidateBox.Intersection(&existingBox) > nmsThreshold {
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
// session, err := InitDFINESession("dfine.onnx", 640, 640, []int{8, 16, 32}, []int{512, 1024,
// 2048})
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
		e = common.ExtractMultiScaleFeatures(
			pic,
			modelSession.FeatStrides,
			modelSession.FeatChannels[0],
			modelSession.FeatureMaps[0].GetData(),
		)
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

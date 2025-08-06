package onnx

import (
	"fmt"
	"image"
	"log"
	"os"
	"sort"
	"sync"
	"time"

	"gocv.io/x/gocv"
)

// FastRCNNClasses for FastRCNN models (COCO classes)
var FastRCNNClasses = []string{
	"__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
	"frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
	"bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// FastRCNNDetection represents a detected object from FastRCNN
type FastRCNNDetection struct {
	Box       image.Rectangle
	Score     float32
	ClassID   int
	ClassName string
}

// FastRCNNModel handles FastRCNN model inference using gocv.ReadNet()
type FastRCNNModel struct {
	modelPath           string
	inputShape          image.Point
	confidenceThreshold float32
	nmsThreshold        float32
	relevantClasses     map[string]bool
	initialized         bool
	mu                  sync.RWMutex
	net                 gocv.Net
	outputNames         []string
}

// FastRCNNConfig for FastRCNN model
type FastRCNNConfig struct {
	ModelPath           string
	InputShape          image.Point
	ConfidenceThreshold float32
	NMSThreshold        float32
	RelevantClasses     []string
}

// NewFastRCNNModel creates a new FastRCNN model detector
func NewFastRCNNModel(config FastRCNNConfig) (*FastRCNNModel, error) {
	detector := &FastRCNNModel{
		modelPath:           config.ModelPath,
		inputShape:          config.InputShape,
		confidenceThreshold: config.ConfidenceThreshold,
		nmsThreshold:        config.NMSThreshold,
		relevantClasses:     make(map[string]bool),
	}

	// Set up relevant classes
	for _, className := range config.RelevantClasses {
		detector.relevantClasses[className] = true
	}

	// Initialize FastRCNN runtime
	if err := detector.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize FastRCNN model: %w", err)
	}

	return detector, nil
}

// initialize sets up the FastRCNN runtime environment using gocv.ReadNet()
func (d *FastRCNNModel) initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Check if model file exists
	if _, err := os.Stat(d.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("FastRCNN model file not found: %s", d.modelPath)
	}

	// Check file size to ensure it's not empty
	fileInfo, err := os.Stat(d.modelPath)
	if err != nil {
		return fmt.Errorf("failed to get FastRCNN model file info: %w", err)
	}
	if fileInfo.Size() == 0 {
		return fmt.Errorf("FastRCNN model file is empty: %s", d.modelPath)
	}

	// Load the FastRCNN model using gocv.ReadNet() with error handling
	net := gocv.ReadNet(d.modelPath, "")

	// Check if net is nil or empty
	if net.Empty() {
		return fmt.Errorf("failed to load FastRCNN model: %s (model may be incompatible with OpenCV DNN)", d.modelPath)
	}

	// Validate the network before proceeding
	if net.GetLayerNames() == nil {
		net.Close()
		return fmt.Errorf("failed to get layer names from FastRCNN model: %s", d.modelPath)
	}

	d.net = net

	// Set backend and target (CPU for now, can be extended for GPU)
	d.net.SetPreferableBackend(gocv.NetBackendOpenCV)
	d.net.SetPreferableTarget(gocv.NetTargetCPU)

	// Get output layer names with validation
	d.outputNames = d.net.GetLayerNames()
	if len(d.outputNames) == 0 {
		d.net.Close()
		return fmt.Errorf("failed to get output layer names from FastRCNN model: %s", d.modelPath)
	}

	d.initialized = true
	log.Printf("✅ FastRCNN model initialized with model: %s", d.modelPath)
	log.Printf("📋 Input shape: %dx%d", d.inputShape.X, d.inputShape.Y)
	log.Printf("🎯 Confidence threshold: %.2f", d.confidenceThreshold)
	log.Printf("🔍 Relevant classes: %v", d.GetRelevantClasses())
	log.Printf("📊 Output layers: %v", d.outputNames)

	return nil
}

// Detect runs inference on the input image using FastRCNN
func (d *FastRCNNModel) Detect(img gocv.Mat) ([]FastRCNNDetection, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("FastRCNN model not initialized")
	}

	// Add timeout protection for inference
	done := make(chan []FastRCNNDetection, 1)
	errChan := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				errChan <- fmt.Errorf("panic during FastRCNN inference: %v", r)
			}
		}()

		// Preprocess the image for FastRCNN
		blob := d.preprocessImage(img)
		defer blob.Close()

		// Run inference with error checking
		outputs := d.net.Forward("")
		if outputs.Empty() {
			errChan <- fmt.Errorf("FastRCNN inference returned empty output")
			return
		}
		defer outputs.Close()

		// Postprocess the outputs for FastRCNN
		size := img.Size()
		detections := d.postprocessOutputs(outputs, image.Point{X: size[1], Y: size[0]})
		done <- detections
	}()

	// Wait for inference with timeout
	select {
	case detections := <-done:
		return detections, nil
	case err := <-errChan:
		return nil, err
	case <-time.After(5 * time.Second): // 5 second timeout
		return nil, fmt.Errorf("FastRCNN inference timeout after 5 seconds")
	}
}

// DetectROI runs inference on a specific region of interest using FastRCNN
func (d *FastRCNNModel) DetectROI(img gocv.Mat, roi image.Rectangle) ([]FastRCNNDetection, error) {
	// Extract the ROI from the image
	roiMat := img.Region(roi)
	defer roiMat.Close()

	// Run detection on the ROI
	detections, err := d.Detect(roiMat)
	if err != nil {
		return nil, err
	}

	// Adjust bounding box coordinates to original image space
	for i := range detections {
		detections[i].Box = detections[i].Box.Add(roi.Min)
	}

	return detections, nil
}

// preprocessImage prepares the input image for FastRCNN model
func (d *FastRCNNModel) preprocessImage(img gocv.Mat) gocv.Mat {
	// Resize image to model input size
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, d.inputShape, 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	// FastRCNN specific preprocessing
	// Convert to blob with FastRCNN normalization
	blob := gocv.BlobFromImage(resized, 1.0, d.inputShape, gocv.NewScalar(0, 0, 0, 0), true, false)

	return blob
}

// postprocessOutputs processes the FastRCNN model outputs to extract detections
func (d *FastRCNNModel) postprocessOutputs(outputs gocv.Mat, originalSize image.Point) []FastRCNNDetection {
	var detections []FastRCNNDetection

	// FastRCNN output format: [batch, num_detections, 7]
	// Each detection: [image_id, label, confidence, x1, y1, x2, y2]
	rows := outputs.Rows()
	cols := outputs.Cols()

	// Process each detection
	for i := 0; i < rows; i++ {
		// Get confidence score
		confidence := outputs.GetFloatAt(i, 2)
		if confidence < d.confidenceThreshold {
			continue
		}

		// Get class ID
		classID := int(outputs.GetFloatAt(i, 1))
		if classID <= 0 || classID >= len(FastRCNNClasses) {
			continue
		}

		// Get bounding box coordinates
		x1 := int(outputs.GetFloatAt(i, 3) * float32(originalSize.X))
		y1 := int(outputs.GetFloatAt(i, 4) * float32(originalSize.Y))
		x2 := int(outputs.GetFloatAt(i, 5) * float32(originalSize.X))
		y2 := int(outputs.GetFloatAt(i, 6) * float32(originalSize.Y))

		// Ensure coordinates are within image bounds
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(originalSize.X, x2)
		y2 = min(originalSize.Y, y2)

		// Skip if bounding box is too small
		if x2-x1 < 10 || y2-y1 < 10 {
			continue
		}

		// Create detection
		detection := FastRCNNDetection{
			Box:       image.Rect(x1, y1, x2, y2),
			Score:     confidence,
			ClassID:   classID,
			ClassName: d.getClassByID(classID),
		}

		detections = append(detections, detection)
	}

	// Apply Non-Maximum Suppression
	detections = d.applyNMS(detections)

	return detections
}

// applyNMS applies Non-Maximum Suppression to remove overlapping detections
func (d *FastRCNNModel) applyNMS(detections []FastRCNNDetection) []FastRCNNDetection {
	if len(detections) == 0 {
		return detections
	}

	// Sort by confidence score (descending)
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Score > detections[j].Score
	})

	var result []FastRCNNDetection
	used := make([]bool, len(detections))

	for i := 0; i < len(detections); i++ {
		if used[i] {
			continue
		}

		result = append(result, detections[i])
		used[i] = true

		// Check overlap with remaining detections
		for j := i + 1; j < len(detections); j++ {
			if used[j] {
				continue
			}

			// Calculate IoU
			iou := calculateIoU(detections[i].Box, detections[j].Box)
			if iou > d.nmsThreshold {
				used[j] = true
			}
		}
	}

	return result
}

// getClassByID returns the class name for a given class ID
func (d *FastRCNNModel) getClassByID(classID int) string {
	if classID >= 0 && classID < len(FastRCNNClasses) {
		return FastRCNNClasses[classID]
	}
	return fmt.Sprintf("unknown_%d", classID)
}

// IsRelevantClass checks if a class is in the relevant classes list
func (d *FastRCNNModel) IsRelevantClass(className string) bool {
	return d.relevantClasses[className]
}

// GetRelevantClasses returns the list of relevant classes
func (d *FastRCNNModel) GetRelevantClasses() []string {
	var classes []string
	for class := range d.relevantClasses {
		classes = append(classes, class)
	}
	return classes
}

// Close releases resources
func (d *FastRCNNModel) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.net.Empty() {
		d.net.Close()
	}
	d.initialized = false
	log.Printf("🔒 FastRCNN model closed")
}

// GetModelInfo returns information about the loaded FastRCNN model
func (d *FastRCNNModel) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model_path":           d.modelPath,
		"input_shape":          d.inputShape,
		"confidence_threshold": d.confidenceThreshold,
		"nms_threshold":        d.nmsThreshold,
		"relevant_classes":     d.GetRelevantClasses(),
		"initialized":          d.initialized,
		"output_layers":        d.outputNames,
		"model_type":           "FastRCNN",
	}
}

// ValidateModel checks if the FastRCNN model is valid and accessible
func (d *FastRCNNModel) ValidateModel() error {
	if _, err := os.Stat(d.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("FastRCNN model file not found: %s", d.modelPath)
	}

	// Additional validation can be added here
	// - Check file size
	// - Validate ONNX format
	// - Check model metadata

	return nil
}

// GetFastRCNNClasses returns the FastRCNN class names
func GetFastRCNNClasses() []string {
	return FastRCNNClasses
}

// GetFastRCNNClassMapping returns a mapping of class names to their IDs
func GetFastRCNNClassMapping() map[string]int {
	mapping := make(map[string]int)
	for i, className := range FastRCNNClasses {
		mapping[className] = i
	}
	return mapping
}

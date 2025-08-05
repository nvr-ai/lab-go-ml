package onnx

import (
	"fmt"
	"image"
	"log"
	"os"
	"sort"
	"sync"

	"gocv.io/x/gocv"
)


// Detection represents a detected object
type Detection struct {
	Box       image.Rectangle
	Score     float32
	ClassID   int
	ClassName string
}

// ONNXDetector handles ONNX model inference using gocv.ReadNet()
type ONNXDetector struct {
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


// NewONNXDetector creates a new ONNX detector
func NewONNXDetector(config Config) (*ONNXDetector, error) {
	detector := & ONNXDetector{
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

	// Initialize ONNX runtime
	if err := detector.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX detector: %w", err)
	}

	return detector, nil
}

// initialize sets up the ONNX runtime environment using gocv.ReadNet()
func (d *ONNXDetector) initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Check if model file exists
	if _, err := os.Stat(d.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", d.modelPath)
	}

	// Load the ONNX model using gocv.ReadNet()
	net := gocv.ReadNet(d.modelPath, "")
	if net.Empty() {
		return fmt.Errorf("failed to load ONNX model: %s", d.modelPath)
	}
	d.net = net

	// Set backend and target (CPU for now, can be extended for GPU)
	d.net.SetPreferableBackend(gocv.NetBackendOpenCV)
	d.net.SetPreferableTarget(gocv.NetTargetCPU)

	// Get output layer names
	d.outputNames = d.net.GetLayerNames()
	if len(d.outputNames) == 0 {
		return fmt.Errorf("failed to get output layer names from model")
	}

	d.initialized = true
	log.Printf("✅ ONNX detector initialized with model: %s", d.modelPath)
	log.Printf("📋 Input shape: %dx%d", d.inputShape.X, d.inputShape.Y)
	log.Printf("🎯 Confidence threshold: %.2f", d.confidenceThreshold)
	log.Printf("🔍 Relevant classes: %v", d.GetRelevantClasses())
	log.Printf("📊 Output layers: %v", d.outputNames)

	return nil
}

// Detect runs inference on the input image
func (d *ONNXDetector) Detect(img gocv.Mat) ([]Detection, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("detector not initialized")
	}

	// Preprocess the image
	blob := d.preprocessImage(img)
	defer blob.Close()

	// Run inference
	outputs := d.net.Forward("")
	defer outputs.Close()

	// Postprocess the outputs
	size := img.Size()
	detections := d.postprocessOutputs(outputs, image.Point{X: size[1], Y: size[0]})

	return detections, nil
}

// DetectROI runs inference on a specific region of interest
func (d *ONNXDetector) DetectROI(img gocv.Mat, roi image.Rectangle) ([]Detection, error) {
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

// preprocessImage prepares the input image for the model
func (d *ONNXDetector) preprocessImage(img gocv.Mat) gocv.Mat {
	// Resize image to model input size
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, d.inputShape, 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	// Convert to blob (normalize and change format)
	blob := gocv.BlobFromImage(resized, 1.0/255.0, d.inputShape, gocv.NewScalar(0, 0, 0, 0), true, false)

	return blob
}

// postprocessOutputs processes the model outputs to extract detections
func (d *ONNXDetector) postprocessOutputs(outputs gocv.Mat, originalSize image.Point) []Detection {
	var detections []Detection

	// Get output dimensions
	rows := outputs.Rows()
	cols := outputs.Cols()

	// Process each detection
	for i := 0; i < rows; i++ {
		// Get confidence scores for all classes
		confidence := outputs.GetFloatAt(i, 4)
		if confidence < d.confidenceThreshold {
			continue
		}

		// Find the class with highest confidence
		classID := 0
		maxScore := float32(0)
		for j := 5; j < cols; j++ {
			score := outputs.GetFloatAt(i, j)
			if score > maxScore {
				maxScore = score
				classID = j - 5
			}
		}

		// Calculate final confidence
		finalConfidence := confidence * maxScore
		if finalConfidence < d.confidenceThreshold {
			continue
		}

		// Get bounding box coordinates
		centerX := outputs.GetFloatAt(i, 0)
		centerY := outputs.GetFloatAt(i, 1)
		width := outputs.GetFloatAt(i, 2)
		height := outputs.GetFloatAt(i, 3)

		// Convert normalized coordinates to pixel coordinates
		x1 := int((centerX - width/2) * float32(originalSize.X))
		y1 := int((centerY - height/2) * float32(originalSize.Y))
		x2 := int((centerX + width/2) * float32(originalSize.X))
		y2 := int((centerY + height/2) * float32(originalSize.Y))

		// Ensure coordinates are within image bounds
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(originalSize.X, x2)
		y2 = min(originalSize.Y, y2)

		// Create detection
		detection := Detection{
			Box:       image.Rect(x1, y1, x2, y2),
			Score:     finalConfidence,
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
func (d *ONNXDetector) applyNMS(detections []Detection) []Detection {
	if len(detections) == 0 {
		return detections
	}

	// Sort by confidence score (descending)
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Score > detections[j].Score
	})

	var result []Detection
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

// calculateIoU calculates the Intersection over Union between two rectangles
func calculateIoU(box1, box2 image.Rectangle) float32 {
	// Calculate intersection
	x1 := max(box1.Min.X, box2.Min.X)
	y1 := max(box1.Min.Y, box2.Min.Y)
	x2 := min(box1.Max.X, box2.Max.X)
	y2 := min(box1.Max.Y, box2.Max.Y)

	if x2 <= x1 || y2 <= y1 {
		return 0.0
	}

	intersection := (x2 - x1) * (y2 - y1)

	// Calculate union
	area1 := (box1.Max.X - box1.Min.X) * (box1.Max.Y - box1.Min.Y)
	area2 := (box2.Max.X - box2.Min.X) * (box2.Max.Y - box2.Min.Y)
	union := area1 + area2 - intersection

	return float32(intersection) / float32(union)
}

// getClassByID returns the class name for a given class ID
func (d *ONNXDetector) getClassByID(classID int) string {
	if classID >= 0 && classID < len(COCOClasses) {
		return COCOClasses[classID]
	}
	return fmt.Sprintf("unknown_%d", classID)
}

// IsRelevantClass checks if a class is in the relevant classes list
func (d *ONNXDetector) IsRelevantClass(className string) bool {
	return d.relevantClasses[className]
}

// GetRelevantClasses returns the list of relevant classes
func (d *ONNXDetector) GetRelevantClasses() []string {
	var classes []string
	for class := range d.relevantClasses {
		classes = append(classes, class)
	}
	return classes
}

// Close releases resources
func (d *ONNXDetector) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.net.Empty() {
		d.net.Close()
	}
	d.initialized = false
	log.Printf("🔒 ONNX detector closed")
}

// GetModelInfo returns information about the loaded model
func (d *ONNXDetector) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model_path":           d.modelPath,
		"input_shape":          d.inputShape,
		"confidence_threshold": d.confidenceThreshold,
		"nms_threshold":        d.nmsThreshold,
		"relevant_classes":     d.GetRelevantClasses(),
		"initialized":          d.initialized,
		"output_layers":        d.outputNames,
	}
}

// ValidateModel checks if the ONNX model is valid and accessible
func (d *ONNXDetector) ValidateModel() error {
	if _, err := os.Stat(d.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", d.modelPath)
	}

	// Additional validation can be added here
	// - Check file size
	// - Validate ONNX format
	// - Check model metadata

	return nil
}

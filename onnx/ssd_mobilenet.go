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

// SSDModel handles SSD MobileNet v2 model inference using gocv.ReadNet()
type SSDModel struct {
	modelPath           string
	configPath          string
	inputShape          image.Point
	confidenceThreshold float32
	nmsThreshold        float32
	relevantClasses     map[string]bool
	initialized         bool
	mu                  sync.RWMutex
	net                 gocv.Net
	outputNames         []string
	modelType           string // "tensorflow" or "onnx"
}

// SSDConfig for SSD MobileNet v2 model
type SSDConfig struct {
	ModelPath           string
	ConfigPath          string // For TensorFlow .pbtxt files
	InputShape          image.Point
	ConfidenceThreshold float32
	NMSThreshold        float32
	RelevantClasses     []string
	ModelType           string // "tensorflow" or "onnx"
}

// NewSSDModel creates a new SSD MobileNet v2 model detector
func NewSSDModel(config SSDConfig) (*SSDModel, error) {
	detector := &SSDModel{
		modelPath:           config.ModelPath,
		configPath:          config.ConfigPath,
		inputShape:          config.InputShape,
		confidenceThreshold: config.ConfidenceThreshold,
		nmsThreshold:        config.NMSThreshold,
		relevantClasses:     make(map[string]bool),
		modelType:           config.ModelType,
	}

	// Set up relevant classes
	for _, className := range config.RelevantClasses {
		detector.relevantClasses[className] = true
	}

	// If this is a TensorFlow model, try to parse the config and generate .pbtxt
	if config.ModelType == "tensorflow" {
		// For TensorFlow models, we'll use default parameters
		// since the config parsing functions were removed
		fmt.Printf("ℹ️  TensorFlow model detected, using default parameters\n")
	}

	// Initialize SSD runtime
	if err := detector.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize SSD MobileNet v2 model: %w", err)
	}

	return detector, nil
}

// initialize sets up the SSD MobileNet v2 runtime environment using gocv.ReadNet()
func (d *SSDModel) initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Check if model file exists
	if _, err := os.Stat(d.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("SSD MobileNet v2 model file not found: %s", d.modelPath)
	}

	// Check file size to ensure it's not empty
	fileInfo, err := os.Stat(d.modelPath)
	if err != nil {
		return fmt.Errorf("failed to get SSD MobileNet v2 model file info: %w", err)
	}
	if fileInfo.Size() == 0 {
		return fmt.Errorf("SSD MobileNet v2 model file is empty: %s", d.modelPath)
	}

	// Add panic recovery for model loading
	var net gocv.Net
	var loadErr error

	func() {
		defer func() {
			if r := recover(); r != nil {
				loadErr = fmt.Errorf("panic during SSD MobileNet v2 model loading: %v", r)
				log.Printf("⚠️  Panic during SSD MobileNet v2 model loading: %v", r)
			}
		}()

		// Load the model based on type
		if d.modelType == "tensorflow" {
			// Load TensorFlow model with config file
			if d.configPath != "" {
				if _, err := os.Stat(d.configPath); os.IsNotExist(err) {
					return
				}
				net = gocv.ReadNet(d.modelPath, d.configPath)
			} else {
				net = gocv.ReadNet(d.modelPath, "")
			}
		} else if d.modelType == "onnx" {
			// Load ONNX model using gocv.ReadNet()
			// For ONNX models, we pass the model path and empty string for config
			net = gocv.ReadNet(d.modelPath, "")
			log.Printf("🔄 Loading ONNX model: %s", d.modelPath)
		} else {
			// Default to ONNX format
			net = gocv.ReadNet(d.modelPath, "")
			log.Printf("🔄 Loading model as ONNX: %s", d.modelPath)
		}
	}()

	if loadErr != nil {
		return loadErr
	}

	// Check if net is nil or empty with additional validation
	if net.Empty() {
		return fmt.Errorf("failed to load SSD MobileNet v2 model: %s (model may be incompatible with OpenCV DNN)", d.modelPath)
	}

	// Validate the network before proceeding with additional checks
	var layerNames []string
	func() {
		defer func() {
			if r := recover(); r != nil {
				loadErr = fmt.Errorf("panic during layer name retrieval: %v", r)
				log.Printf("⚠️  Panic during layer name retrieval: %v", r)
			}
		}()
		layerNames = net.GetLayerNames()
	}()

	if loadErr != nil {
		net.Close()
		return loadErr
	}

	if layerNames == nil || len(layerNames) == 0 {
		net.Close()
		return fmt.Errorf("failed to get layer names from SSD MobileNet v2 model: %s", d.modelPath)
	}

	d.net = net

	// Set backend and target (CPU for now, can be extended for GPU)
	d.net.SetPreferableBackend(gocv.NetBackendOpenCV)
	d.net.SetPreferableTarget(gocv.NetTargetCPU)

	// Get output layer names with validation
	d.outputNames = d.net.GetLayerNames()
	if len(d.outputNames) == 0 {
		d.net.Close()
		return fmt.Errorf("failed to get output layer names from SSD MobileNet v2 model: %s", d.modelPath)
	}

	d.initialized = true
	log.Printf("✅ SSD MobileNet v2 model initialized with model: %s", d.modelPath)
	log.Printf("📋 Input shape: %dx%d", d.inputShape.X, d.inputShape.Y)
	log.Printf("🎯 Confidence threshold: %.2f", d.confidenceThreshold)
	log.Printf("🔍 Relevant classes: %v", d.GetRelevantClasses())
	log.Printf("📊 Output layers: %v", d.outputNames)
	log.Printf("🏷️  Model type: %s", d.modelType)

	return nil
}

// Detect runs inference on the input image using SSD MobileNet v2
func (d *SSDModel) Detect(img gocv.Mat) ([]Detection, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("SSD MobileNet v2 model not initialized")
	}

	// Add timeout protection for inference
	done := make(chan []Detection, 1)
	errChan := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				errChan <- fmt.Errorf("panic during SSD MobileNet v2 inference: %v", r)
			}
		}()

		// Preprocess the image for SSD MobileNet v2
		blob := d.preprocessImage(img)
		defer blob.Close()

		// Set input blob
		d.net.SetInput(blob, "")

		// Run inference with error checking
		outputs := d.net.Forward("")
		if outputs.Empty() {
			errChan <- fmt.Errorf("SSD MobileNet v2 inference returned empty output")
			return
		}
		defer outputs.Close()

		// Postprocess the outputs for SSD MobileNet v2
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
		return nil, fmt.Errorf("SSD MobileNet v2 inference timeout after 5 seconds")
	}
}

// DetectROI runs inference on a specific region of interest using SSD MobileNet v2
func (d *SSDModel) DetectROI(img gocv.Mat, roi image.Rectangle) ([]Detection, error) {
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

// preprocessImage prepares the input image for SSD MobileNet v2 model
func (d *SSDModel) preprocessImage(img gocv.Mat) gocv.Mat {
	// Resize image to model input size (320x320 for SSD MobileNet v2 FPNLite)
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, d.inputShape, 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	// SSD MobileNet v2 specific preprocessing
	if d.modelType == "tensorflow" {
		// TensorFlow models: normalize to [-1, 1] range
		// (pixel - 127.5) / 127.5
		blob := gocv.BlobFromImage(resized, 1.0/127.5, d.inputShape, gocv.NewScalar(127.5, 127.5, 127.5, 0), true, false)
		return blob
	} else {
		// ONNX models: typically normalize to [0, 1] range
		// For SSD MobileNet v2 FPNLite 320x320 ONNX, we usually use:
		// - Input size: 320x320
		// - Normalization: pixel / 255.0 (scale to [0, 1])
		// - SwapRB: true (BGR to RGB conversion)
		// - Crop: false (no center crop)
		
		// For ONNX models, we typically use simple normalization to [0, 1]
		blob := gocv.BlobFromImage(resized, 1.0/255.0, d.inputShape, gocv.NewScalar(0, 0, 0, 0), true, false)
		return blob
	}
}

// postprocessOutputs processes the SSD MobileNet v2 model outputs to extract detections
func (d *SSDModel) postprocessOutputs(outputs gocv.Mat, originalSize image.Point) []Detection {
	var detections []Detection

	// SSD MobileNet v2 output format varies by model type
	if d.modelType == "tensorflow" {
		// TensorFlow format: Multiple output tensors
		// - detection_boxes: [1, N, 4] - normalized coordinates [y1, x1, y2, x2]
		// - detection_scores: [1, N] - confidence scores
		// - detection_classes: [1, N] - class IDs
		// - num_detections: [1] - number of detections
		
		rows := outputs.Rows()
		cols := outputs.Cols()
		
		// Handle different output formats
		if rows == 1 && cols > 7 {
			// Single row with multiple detections (legacy format)
			numDetections := cols / 7
			for i := 0; i < numDetections; i++ {
				confidence := outputs.GetFloatAt(0, i*7+2)
				if confidence < d.confidenceThreshold {
					continue
				}

				classID := int(outputs.GetFloatAt(0, i*7+1))
				if classID <= 0 || classID >= len(COCOClasses) {
					continue
				}

				// Get normalized coordinates (TensorFlow format: [y1, x1, y2, x2])
				yMin := outputs.GetFloatAt(0, i*7+3)
				xMin := outputs.GetFloatAt(0, i*7+4)
				yMax := outputs.GetFloatAt(0, i*7+5)
				xMax := outputs.GetFloatAt(0, i*7+6)

				// Scale to original image size
				x1 := int(xMin * float32(originalSize.X))
				y1 := int(yMin * float32(originalSize.Y))
				x2 := int(xMax * float32(originalSize.X))
				y2 := int(yMax * float32(originalSize.Y))

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
				detection := Detection{
					Box:       image.Rect(x1, y1, x2, y2),
					Score:     confidence,
					ClassID:   classID,
					ClassName: d.getClassByID(classID),
				}

				detections = append(detections, detection)
			}
		} else {
			// Multiple rows format or separate output tensors
			for i := 0; i < rows; i++ {
				confidence := outputs.GetFloatAt(i, 2)
				if confidence < d.confidenceThreshold {
					continue
				}

				classID := int(outputs.GetFloatAt(i, 1))
				if classID <= 0 || classID >= len(COCOClasses) {
					continue
				}

				// Get normalized coordinates (TensorFlow format: [y1, x1, y2, x2])
				yMin := outputs.GetFloatAt(i, 3)
				xMin := outputs.GetFloatAt(i, 4)
				yMax := outputs.GetFloatAt(i, 5)
				xMax := outputs.GetFloatAt(i, 6)

				// Scale to original image size
				x1 := int(xMin * float32(originalSize.X))
				y1 := int(yMin * float32(originalSize.Y))
				x2 := int(xMax * float32(originalSize.X))
				y2 := int(yMax * float32(originalSize.Y))

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
				detection := Detection{
					Box:       image.Rect(x1, y1, x2, y2),
					Score:     confidence,
					ClassID:   classID,
					ClassName: d.getClassByID(classID),
				}

				detections = append(detections, detection)
			}
		}
	} else {
		// ONNX format: typically [1, N, 7] where N is number of detections
		// Each detection: [image_id, label, confidence, x_min, y_min, x_max, y_max]
		// Or separate outputs for boxes, scores, and classes
		// For SSD MobileNet v2 FPNLite 320x320, output format is typically:
		// - detection_boxes: [1, N, 4] - normalized coordinates [y1, x1, y2, x2]
		// - detection_scores: [1, N] - confidence scores
		// - detection_classes: [1, N] - class IDs
		
		rows := outputs.Rows()
		cols := outputs.Cols()
		
		log.Printf("🔍 ONNX output shape: %dx%d", rows, cols)

		// Handle different ONNX output formats
		if rows == 1 && cols > 7 {
			// Single row with multiple detections [1, N*7]
			numDetections := cols / 7
			for i := 0; i < numDetections; i++ {
				confidence := outputs.GetFloatAt(0, i*7+2)
				if confidence < d.confidenceThreshold {
					continue
				}

				classID := int(outputs.GetFloatAt(0, i*7+1))
				if classID <= 0 || classID >= len(COCOClasses) {
					continue
				}

				// Get normalized coordinates (ONNX format: [x_min, y_min, x_max, y_max])
				// For SSD MobileNet v2 FPNLite, coordinates are typically [y1, x1, y2, x2]
				yMin := outputs.GetFloatAt(0, i*7+3)
				xMin := outputs.GetFloatAt(0, i*7+4)
				yMax := outputs.GetFloatAt(0, i*7+5)
				xMax := outputs.GetFloatAt(0, i*7+6)

				// Scale to original image size
				x1 := int(xMin * float32(originalSize.X))
				y1 := int(yMin * float32(originalSize.Y))
				x2 := int(xMax * float32(originalSize.X))
				y2 := int(yMax * float32(originalSize.Y))

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
				detection := Detection{
					Box:       image.Rect(x1, y1, x2, y2),
					Score:     confidence,
					ClassID:   classID,
					ClassName: d.getClassByID(classID),
				}

				detections = append(detections, detection)
			}
		} else if rows > 1 && cols == 7 {
			// Multiple rows format [N, 7]
			for i := 0; i < rows; i++ {
				confidence := outputs.GetFloatAt(i, 2)
				if confidence < d.confidenceThreshold {
					continue
				}

				classID := int(outputs.GetFloatAt(i, 1))
				if classID <= 0 || classID >= len(COCOClasses) {
					continue
				}

				// Get normalized coordinates (ONNX format: [x_min, y_min, x_max, y_max])
				// For SSD MobileNet v2 FPNLite, coordinates are typically [y1, x1, y2, x2]
				yMin := outputs.GetFloatAt(i, 3)
				xMin := outputs.GetFloatAt(i, 4)
				yMax := outputs.GetFloatAt(i, 5)
				xMax := outputs.GetFloatAt(i, 6)

				// Scale to original image size
				x1 := int(xMin * float32(originalSize.X))
				y1 := int(yMin * float32(originalSize.Y))
				x2 := int(xMax * float32(originalSize.X))
				y2 := int(yMax * float32(originalSize.Y))

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
				detection := Detection{
					Box:       image.Rect(x1, y1, x2, y2),
					Score:     confidence,
					ClassID:   classID,
					ClassName: d.getClassByID(classID),
				}

				detections = append(detections, detection)
			}
		} else {
			// Try to handle other ONNX output formats
			log.Printf("⚠️  Unknown ONNX output format: %dx%d", rows, cols)
		}
	}

	// Apply Non-Maximum Suppression
	detections = d.applyNMS(detections)

	return detections
}

// applyNMS applies Non-Maximum Suppression to remove overlapping detections
func (d *SSDModel) applyNMS(detections []Detection) []Detection {
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

// getClassByID returns the class name for a given class ID
func (d *SSDModel) getClassByID(classID int) string {
	if classID >= 0 && classID < len(COCOClasses) {
		return COCOClasses[classID]
	}
	return fmt.Sprintf("unknown_%d", classID)
}

// IsRelevantClass checks if a class is in the relevant classes list
func (d *SSDModel) IsRelevantClass(className string) bool {
	return d.relevantClasses[className]
}

// GetRelevantClasses returns the list of relevant classes
func (d *SSDModel) GetRelevantClasses() []string {
	var classes []string
	for class := range d.relevantClasses {
		classes = append(classes, class)
	}
	return classes
}

// Close releases resources
func (d *SSDModel) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()

	if !d.net.Empty() {
		d.net.Close()
	}
	d.initialized = false
	log.Printf("🔒 SSD MobileNet v2 model closed")
}

// GetModelInfo returns information about the loaded SSD MobileNet v2 model
func (d *SSDModel) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model_path":           d.modelPath,
		"config_path":          d.configPath,
		"input_shape":          d.inputShape,
		"confidence_threshold": d.confidenceThreshold,
		"nms_threshold":        d.nmsThreshold,
		"relevant_classes":     d.GetRelevantClasses(),
		"initialized":          d.initialized,
		"output_layers":        d.outputNames,
		"model_type":           d.modelType,
	}
}

// ValidateModel checks if the SSD MobileNet v2 model is valid and accessible
func (d *SSDModel) ValidateModel() error {
	if _, err := os.Stat(d.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("SSD MobileNet v2 model file not found: %s", d.modelPath)
	}

	if d.modelType == "tensorflow" && d.configPath != "" {
		if _, err := os.Stat(d.configPath); os.IsNotExist(err) {
			return fmt.Errorf("SSD MobileNet v2 config file not found: %s", d.configPath)
		}
	}

	return nil
} 
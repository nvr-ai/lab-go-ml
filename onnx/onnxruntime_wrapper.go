package onnx

/*
#cgo CFLAGS: -I${SRCDIR}/../onnxruntimelinux/include
#cgo LDFLAGS: -L${SRCDIR}/../onnxruntimelinux/lib -lonnxruntime
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>

// Helper function to get API
const OrtApi* getOrtApi() {
    return OrtGetApiBase()->GetApi(ORT_API_VERSION);
}
*/
import "C"
import (
	"fmt"
	"image"
	"log"
	"os"
	"sync"
	"unsafe"

	"gocv.io/x/gocv"
)

// ONNXRuntimeModel represents an ONNX Runtime model session
type ONNXRuntimeModel struct {
	env           *C.OrtEnv
	session       *C.OrtSession
	sessionOptions *C.OrtSessionOptions
	allocator     *C.OrtAllocator
	api           *C.OrtApi
	modelPath     string
	inputShape    image.Point
	outputNames   []string
	inputNames    []string
	initialized   bool
	mu            sync.RWMutex
}

// ONNXRuntimeConfig holds configuration for ONNX Runtime model
type ONNXRuntimeConfig struct {
	ModelPath           string
	InputShape          image.Point
	ConfidenceThreshold float32
	NMSThreshold        float32
	RelevantClasses     []string
	ExecutionProvider   string
	GraphOptimizationLevel int
}

// NewONNXRuntimeModel creates a new ONNX Runtime model
func NewONNXRuntimeModel(config ONNXRuntimeConfig) (*ONNXRuntimeModel, error) {
	model := &ONNXRuntimeModel{
		modelPath: config.ModelPath,
		inputShape: config.InputShape,
	}

	if err := model.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime model: %w", err)
	}

	return model, nil
}

// initialize sets up the ONNX Runtime environment
func (m *ONNXRuntimeModel) initialize() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if model file exists
	if _, err := os.Stat(m.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("ONNX model file not found: %s", m.modelPath)
	}

	// Get ONNX Runtime API
	m.api = C.getOrtApi()
	if m.api == nil {
		return fmt.Errorf("failed to get ONNX Runtime API")
	}

	// Create environment
	var env *C.OrtEnv
	status := m.api.CreateEnv(C.ORT_LOGGING_LEVEL_WARNING, C.CString("onnxruntime"), &env)
	if status != nil {
		return fmt.Errorf("failed to create ONNX Runtime environment: %s", C.GoString(m.api.GetErrorMessage(status)))
	}
	m.env = env

	// Create session options
	var sessionOptions *C.OrtSessionOptions
	status = m.api.CreateSessionOptions(&sessionOptions)
	if status != nil {
		m.api.ReleaseEnv(m.env)
		return fmt.Errorf("failed to create session options: %s", C.GoString(m.api.GetErrorMessage(status)))
	}
	m.sessionOptions = sessionOptions

	// Set graph optimization level
	graphOptimizationLevel := C.ORT_ENABLE_ALL
	status = m.api.SetSessionGraphOptimizationLevel(m.sessionOptions, graphOptimizationLevel)
	if status != nil {
		m.api.ReleaseSessionOptions(m.sessionOptions)
		m.api.ReleaseEnv(m.env)
		return fmt.Errorf("failed to set graph optimization level: %s", C.GoString(m.api.GetErrorMessage(status)))
	}

	// Create session
	var session *C.OrtSession
	modelPathC := C.CString(m.modelPath)
	defer C.free(unsafe.Pointer(modelPathC))
	
	status = m.api.CreateSession(m.env, modelPathC, m.sessionOptions, &session)
	if status != nil {
		m.api.ReleaseSessionOptions(m.sessionOptions)
		m.api.ReleaseEnv(m.env)
		return fmt.Errorf("failed to create session: %s", C.GoString(m.api.GetErrorMessage(status)))
	}
	m.session = session

	// Get input and output names
	if err := m.getInputOutputNames(); err != nil {
		m.api.ReleaseSession(m.session)
		m.api.ReleaseSessionOptions(m.sessionOptions)
		m.api.ReleaseEnv(m.env)
		return fmt.Errorf("failed to get input/output names: %w", err)
	}

	// Get allocator
	var allocator *C.OrtAllocator
	status = m.api.GetAllocatorWithDefaultOptions(&allocator)
	if status != nil {
		m.api.ReleaseSession(m.session)
		m.api.ReleaseSessionOptions(m.sessionOptions)
		m.api.ReleaseEnv(m.env)
		return fmt.Errorf("failed to get allocator: %s", C.GoString(m.api.GetErrorMessage(status)))
	}
	m.allocator = allocator

	m.initialized = true
	log.Printf("✅ ONNX Runtime model initialized successfully: %s", m.modelPath)
	log.Printf("📋 Input shape: %dx%d", m.inputShape.X, m.inputShape.Y)
	log.Printf("📊 Input names: %v", m.inputNames)
	log.Printf("📊 Output names: %v", m.outputNames)

	return nil
}

// getInputOutputNames retrieves input and output names from the model
func (m *ONNXRuntimeModel) getInputOutputNames() error {
	// Get input count
	var inputCount C.size_t
	status := m.api.SessionGetInputCount(m.session, &inputCount)
	if status != nil {
		return fmt.Errorf("failed to get input count: %s", C.GoString(m.api.GetErrorMessage(status)))
	}

	// Get output count
	var outputCount C.size_t
	status = m.api.SessionGetOutputCount(m.session, &outputCount)
	if status != nil {
		return fmt.Errorf("failed to get output count: %s", C.GoString(m.api.GetErrorMessage(status)))
	}

	// Get input names
	m.inputNames = make([]string, inputCount)
	for i := C.size_t(0); i < inputCount; i++ {
		var name *C.char
		status := m.api.SessionGetInputName(m.session, i, m.allocator, &name)
		if status != nil {
			return fmt.Errorf("failed to get input name %d: %s", i, C.GoString(m.api.GetErrorMessage(status)))
		}
		m.inputNames[i] = C.GoString(name)
		m.api.AllocatorFree(m.allocator, unsafe.Pointer(name))
	}

	// Get output names
	m.outputNames = make([]string, outputCount)
	for i := C.size_t(0); i < outputCount; i++ {
		var name *C.char
		status := m.api.SessionGetOutputName(m.session, i, m.allocator, &name)
		if status != nil {
			return fmt.Errorf("failed to get output name %d: %s", i, C.GoString(m.api.GetErrorMessage(status)))
		}
		m.outputNames[i] = C.GoString(name)
		m.api.AllocatorFree(m.allocator, unsafe.Pointer(name))
	}

	return nil
}

// Detect runs inference on the input image using ONNX Runtime
func (m *ONNXRuntimeModel) Detect(img gocv.Mat) ([]Detection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.initialized {
		return nil, fmt.Errorf("ONNX Runtime model not initialized")
	}

	// Preprocess image
	inputTensor, err := m.preprocessImage(img)
	if err != nil {
		return nil, fmt.Errorf("failed to preprocess image: %w", err)
	}
	defer m.api.ReleaseValue(inputTensor)

	// Prepare input names
	inputNames := make([]*C.char, len(m.inputNames))
	for i, name := range m.inputNames {
		inputNames[i] = C.CString(name)
		defer C.free(unsafe.Pointer(inputNames[i]))
	}

	// Prepare output names
	outputNames := make([]*C.char, len(m.outputNames))
	for i, name := range m.outputNames {
		outputNames[i] = C.CString(name)
		defer C.free(unsafe.Pointer(outputNames[i]))
	}

	// Prepare inputs
	inputs := []*C.OrtValue{inputTensor}

	// Prepare outputs
	outputs := make([]*C.OrtValue, len(m.outputNames))
	for i := range outputs {
		outputs[i] = nil
	}

	// Run inference
	status := m.api.Run(m.session, nil, 
		(**C.char)(unsafe.Pointer(&inputNames[0])), 
		(**C.OrtValue)(unsafe.Pointer(&inputs[0])), 
		C.size_t(len(inputs)),
		(**C.char)(unsafe.Pointer(&outputNames[0])), 
		C.size_t(len(outputNames)),
		(**C.OrtValue)(unsafe.Pointer(&outputs[0])))
	
	if status != nil {
		return nil, fmt.Errorf("failed to run inference: %s", C.GoString(m.api.GetErrorMessage(status)))
	}

	// Process outputs
	detections, err := m.postprocessOutputs(outputs, image.Point{X: img.Cols(), Y: img.Rows()})
	if err != nil {
		return nil, fmt.Errorf("failed to postprocess outputs: %w", err)
	}

	// Clean up outputs
	for _, output := range outputs {
		if output != nil {
			m.api.ReleaseValue(output)
		}
	}

	return detections, nil
}

// preprocessImage prepares the input image for ONNX Runtime
func (m *ONNXRuntimeModel) preprocessImage(img gocv.Mat) (*C.OrtValue, error) {
	// Resize image to model input size
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, m.inputShape, 0, 0, gocv.InterpolationLinear)
	defer resized.Close()

	// Convert to blob (normalize to [0, 1])
	blob := gocv.BlobFromImage(resized, 1.0/255.0, m.inputShape, gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	// Create input tensor
	var inputTensor *C.OrtValue
	status := m.api.CreateTensorWithDataAsOrtValue(
		m.allocator,
		unsafe.Pointer(blob.Ptr()),
		C.size_t(blob.Total()*blob.ElemSize()),
		[]C.int64_t{1, C.int64_t(m.inputShape.Y), C.int64_t(m.inputShape.X), 3},
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&inputTensor)
	
	if status != nil {
		return nil, fmt.Errorf("failed to create input tensor: %s", C.GoString(m.api.GetErrorMessage(status)))
	}

	return inputTensor, nil
}

// postprocessOutputs processes the ONNX Runtime outputs to extract detections
func (m *ONNXRuntimeModel) postprocessOutputs(outputs []*C.OrtValue, originalSize image.Point) ([]Detection, error) {
	var detections []Detection

	// Process the first output (assuming it contains detections)
	if len(outputs) > 0 && outputs[0] != nil {
		// Get tensor info
		var tensorInfo *C.OrtTensorTypeAndShapeInfo
		status := m.api.GetTensorTypeAndShape(outputs[0], &tensorInfo)
		if status != nil {
			return nil, fmt.Errorf("failed to get tensor info: %s", C.GoString(m.api.GetErrorMessage(status)))
		}
		defer m.api.ReleaseTensorTypeAndShapeInfo(tensorInfo)

		// Get tensor data
		var data unsafe.Pointer
		status = m.api.GetTensorMutableData(outputs[0], &data)
		if status != nil {
			return nil, fmt.Errorf("failed to get tensor data: %s", C.GoString(m.api.GetErrorMessage(status)))
		}

		// Get tensor shape
		var dimCount C.size_t
		status = m.api.GetDimensionsCount(tensorInfo, &dimCount)
		if status != nil {
			return nil, fmt.Errorf("failed to get dimensions count: %s", C.GoString(m.api.GetErrorMessage(status)))
		}

		shape := make([]C.int64_t, dimCount)
		status = m.api.GetDimensions(tensorInfo, shape, dimCount)
		if status != nil {
			return nil, fmt.Errorf("failed to get dimensions: %s", C.GoString(m.api.GetErrorMessage(status)))
		}

		// Process detections based on output format
		// For SSD MobileNet v2, output is typically [1, N, 7] where N is number of detections
		if len(shape) >= 2 {
			numDetections := int(shape[1])
			detectionSize := int(shape[len(shape)-1]) // Usually 7 for [image_id, label, confidence, x1, y1, x2, y2]

			// Cast data to float32 array
			floatData := (*[1 << 30]C.float)(data)[:numDetections*detectionSize]

			for i := 0; i < numDetections; i++ {
				offset := i * detectionSize
				if offset+6 >= len(floatData) {
					break
				}

				confidence := float32(floatData[offset+2])
				if confidence < 0.5 { // Confidence threshold
					continue
				}

				classID := int(floatData[offset+1])
				if classID <= 0 || classID >= len(COCOClasses) {
					continue
				}

				// Get normalized coordinates
				x1 := float32(floatData[offset+3])
				y1 := float32(floatData[offset+4])
				x2 := float32(floatData[offset+5])
				y2 := float32(floatData[offset+6])

				// Scale to original image size
				x1Scaled := int(x1 * float32(originalSize.X))
				y1Scaled := int(y1 * float32(originalSize.Y))
				x2Scaled := int(x2 * float32(originalSize.X))
				y2Scaled := int(y2 * float32(originalSize.Y))

				// Ensure coordinates are within image bounds
				x1Scaled = max(0, x1Scaled)
				y1Scaled = max(0, y1Scaled)
				x2Scaled = min(originalSize.X, x2Scaled)
				y2Scaled = min(originalSize.Y, y2Scaled)

				// Skip if bounding box is too small
				if x2Scaled-x1Scaled < 10 || y2Scaled-y1Scaled < 10 {
					continue
				}

				// Create detection
				detection := Detection{
					Box:       image.Rect(x1Scaled, y1Scaled, x2Scaled, y2Scaled),
					Score:     confidence,
					ClassID:   classID,
					ClassName: getClassByID(classID),
				}

				detections = append(detections, detection)
			}
		}
	}

	return detections, nil
}

// Close releases resources
func (m *ONNXRuntimeModel) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initialized {
		if m.session != nil {
			m.api.ReleaseSession(m.session)
		}
		if m.sessionOptions != nil {
			m.api.ReleaseSessionOptions(m.sessionOptions)
		}
		if m.env != nil {
			m.api.ReleaseEnv(m.env)
		}
		m.initialized = false
		log.Printf("🔒 ONNX Runtime model closed")
	}
}

// GetModelInfo returns information about the loaded model
func (m *ONNXRuntimeModel) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model_path":     m.modelPath,
		"input_shape":    m.inputShape,
		"input_names":    m.inputNames,
		"output_names":   m.outputNames,
		"initialized":    m.initialized,
		"runtime":        "ONNX Runtime",
	}
}

// Helper function to get class name by ID
func getClassByID(classID int) string {
	if classID >= 0 && classID < len(COCOClasses) {
		return COCOClasses[classID]
	}
	return fmt.Sprintf("unknown_%d", classID)
} 
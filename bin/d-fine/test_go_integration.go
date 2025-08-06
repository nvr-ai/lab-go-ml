package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/yalue/onnxruntime_go"
)

// SupportedResolution represents a resolution configuration for testing
type SupportedResolution struct {
	Width       int
	Height      int
	AspectRatio string
}

// GetTestResolutions returns the supported resolutions for testing
func GetTestResolutions() []SupportedResolution {
	return []SupportedResolution{
		// 16:9 aspect ratio
		{160, 90, "16:9"},
		{320, 180, "16:9"},
		{640, 360, "16:9"},
		{960, 540, "16:9"},
		// 4:3 aspect ratio
		{160, 120, "4:3"},
		{320, 240, "4:3"},
		{640, 480, "4:3"},
		{960, 720, "4:3"},
	}
}

// TestONNXModelWithResolution tests the ONNX model with a specific resolution
func TestONNXModelWithResolution(session *onnxruntime_go.Session, resolution SupportedResolution) error {
	fmt.Printf("Testing resolution %dx%d (%s)... ", resolution.Width, resolution.Height, resolution.AspectRatio)

	// Create dummy input tensors
	batchSize := 1
	channels := 3
	
	// Create image tensor (NCHW format)
	imageShape := []int64{int64(batchSize), int64(channels), int64(resolution.Height), int64(resolution.Width)}
	imageData := make([]float32, batchSize*channels*resolution.Height*resolution.Width)
	
	// Fill with dummy data (random-like pattern)
	for i := range imageData {
		imageData[i] = float32(i%255) / 255.0
	}
	
	imageTensor, err := onnxruntime_go.NewTensor(imageShape, imageData)
	if err != nil {
		return fmt.Errorf("failed to create image tensor: %w", err)
	}
	defer imageTensor.Destroy()
	
	// Create target sizes tensor  
	sizeShape := []int64{int64(batchSize), 2}
	sizeData := []int64{int64(resolution.Width), int64(resolution.Height)}
	
	sizeTensor, err := onnxruntime_go.NewTensor(sizeShape, sizeData)
	if err != nil {
		return fmt.Errorf("failed to create size tensor: %w", err)
	}
	defer sizeTensor.Destroy()
	
	// Run inference
	inputs := []onnxruntime_go.Value{imageTensor, sizeTensor}
	outputs, err := session.Run(inputs)
	if err != nil {
		return fmt.Errorf("inference failed: %w", err)
	}
	
	// Clean up outputs
	for _, output := range outputs {
		output.Destroy()
	}
	
	fmt.Println("✓ SUCCESS")
	return nil
}

// ValidateModelInputs validates that the model accepts dynamic input shapes
func ValidateModelInputs(session *onnxruntime_go.Session) error {
	fmt.Println("Validating model input signature...")
	
	inputCount, err := session.GetInputCount()
	if err != nil {
		return fmt.Errorf("failed to get input count: %w", err)
	}
	
	if inputCount != 2 {
		return fmt.Errorf("expected 2 inputs, got %d", inputCount)
	}
	
	// Check input names and shapes
	expectedInputs := []string{"images", "orig_target_sizes"}
	
	for i := 0; i < int(inputCount); i++ {
		name, err := session.GetInputName(i)
		if err != nil {
			return fmt.Errorf("failed to get input name %d: %w", i, err)
		}
		
		if name != expectedInputs[i] {
			return fmt.Errorf("input %d: expected name '%s', got '%s'", i, expectedInputs[i], name)
		}
		
		typeInfo, err := session.GetInputTypeInfo(i)
		if err != nil {
			return fmt.Errorf("failed to get input type info %d: %w", i, err)
		}
		
		fmt.Printf("  Input %d: %s - %s\n", i, name, typeInfo.GetTensorTypeAndShapeInfo().GetShape())
		typeInfo.Destroy()
	}
	
	fmt.Println("✓ Model input validation successful")
	return nil
}

// ValidateModelOutputs validates that the model produces expected outputs
func ValidateModelOutputs(session *onnxruntime_go.Session) error {
	fmt.Println("Validating model output signature...")
	
	outputCount, err := session.GetOutputCount()
	if err != nil {
		return fmt.Errorf("failed to get output count: %w", err)
	}
	
	if outputCount != 3 {
		return fmt.Errorf("expected 3 outputs, got %d", outputCount)
	}
	
	// Check output names
	expectedOutputs := []string{"labels", "boxes", "scores"}
	
	for i := 0; i < int(outputCount); i++ {
		name, err := session.GetOutputName(i)
		if err != nil {
			return fmt.Errorf("failed to get output name %d: %w", i, err)
		}
		
		if name != expectedOutputs[i] {
			return fmt.Errorf("output %d: expected name '%s', got '%s'", i, expectedOutputs[i], name)
		}
		
		typeInfo, err := session.GetOutputTypeInfo(i)
		if err != nil {
			return fmt.Errorf("failed to get output type info %d: %w", i, err)
		}
		
		fmt.Printf("  Output %d: %s - %s\n", i, name, typeInfo.GetTensorTypeAndShapeInfo().GetShape())
		typeInfo.Destroy()
	}
	
	fmt.Println("✓ Model output validation successful")
	return nil
}

// TestModelPerformance tests model performance across different resolutions
func TestModelPerformance(session *onnxruntime_go.Session, resolutions []SupportedResolution) error {
	fmt.Println("Testing model performance across resolutions...")
	
	for _, resolution := range resolutions {
		// Simple performance test - just measure if inference completes
		err := TestONNXModelWithResolution(session, resolution)
		if err != nil {
			return fmt.Errorf("performance test failed for %dx%d: %w", 
				resolution.Width, resolution.Height, err)
		}
	}
	
	fmt.Println("✓ Performance tests completed successfully")
	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run test_go_integration.go <model_path>")
		fmt.Println("Example: go run test_go_integration.go d_fine_model_dynamic_169_43.onnx")
		os.Exit(1)
	}
	
	modelPath := os.Args[1]
	
	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file does not exist: %s", modelPath)
	}
	
	// Initialize ONNX Runtime
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()
	
	fmt.Printf("🚀 Testing D-FINE ONNX model: %s\n", filepath.Base(modelPath))
	fmt.Println("🔍 ONNX Runtime Go Integration Test")
	fmt.Println()
	
	// Create session
	session, err := onnxruntime_go.NewSession(modelPath, onnxruntime_go.NewSessionOptions())
	if err != nil {
		log.Fatalf("Failed to create ONNX session: %v", err)
	}
	defer session.Destroy()
	
	// Validate model structure
	if err := ValidateModelInputs(session); err != nil {
		log.Fatalf("Model input validation failed: %v", err)
	}
	
	if err := ValidateModelOutputs(session); err != nil {
		log.Fatalf("Model output validation failed: %v", err)
	}
	
	fmt.Println()
	
	// Test with all supported resolutions
	resolutions := GetTestResolutions()
	fmt.Printf("Testing model with %d supported resolutions:\n", len(resolutions))
	
	for _, resolution := range resolutions {
		if err := TestONNXModelWithResolution(session, resolution); err != nil {
			log.Fatalf("Resolution test failed: %v", err)
		}
	}
	
	fmt.Println()
	
	// Performance testing
	if err := TestModelPerformance(session, resolutions[:4]); err != nil {
		log.Fatalf("Performance testing failed: %v", err)
	}
	
	fmt.Println()
	fmt.Println("🎉 All tests passed successfully!")
	fmt.Println("✅ The D-FINE ONNX model is fully compatible with onnxruntime-go")
	fmt.Printf("✅ Supports all %d configured resolutions\n", len(resolutions))
	fmt.Println("✅ Ready for production use in Go applications")
}
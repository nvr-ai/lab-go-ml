package main

import (
	"fmt"
	"image"
	"log"
	"os"

	"github.com/nvr-ai/go-ml/onnx"
	"gocv.io/x/gocv"
)

func testFastRCNN() {
	fmt.Println("🚀 FastRCNN Model Test")
	fmt.Println("======================")

	// Check if model file exists
	modelPath := "fasterrcnn.onnx"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("❌ Model file not found: %s\n", modelPath)
		fmt.Println("💡 Please make sure you have the FastRCNN model file in the current directory")
		return
	}

	// Initialize FastRCNN model
	fmt.Printf("📁 Loading FastRCNN model: %s\n", modelPath)
	fastRCNN, err := onnx.NewFastRCNNModel(onnx.FastRCNNConfig{
		ModelPath:           modelPath,
		InputShape:          image.Point{X: 800, Y: 600},
		ConfidenceThreshold: 0.5,
		NMSThreshold:        0.3,
		RelevantClasses:     []string{"person", "car", "truck", "bus", "motorcycle", "bicycle"},
	})
	if err != nil {
		log.Fatalf("❌ Failed to initialize FastRCNN model: %v", err)
	}
	defer fastRCNN.Close()

	fmt.Println("✅ FastRCNN model loaded successfully!")

	// Test with a sample image if available
	testImagePath := "test_image.jpg"
	if _, err := os.Stat(testImagePath); err == nil {
		fmt.Printf("🖼️  Testing with image: %s\n", testImagePath)
		
		// Load image
		img := gocv.IMRead(testImagePath, gocv.IMReadColor)
		if img.Empty() {
			fmt.Printf("❌ Failed to load image: %s\n", testImagePath)
			return
		}
		defer img.Close()

		fmt.Printf("📏 Image size: %dx%d\n", img.Cols(), img.Rows())

		// Run detection
		detections, err := fastRCNN.Detect(img)
		if err != nil {
			fmt.Printf("❌ Detection failed: %v\n", err)
			return
		}

		fmt.Printf("🎯 Found %d objects:\n", len(detections))
		for i, detection := range detections {
			if fastRCNN.IsRelevantClass(detection.ClassName) {
				fmt.Printf("  %d. %s (confidence: %.2f) at %v\n", 
					i+1, detection.ClassName, detection.Score, detection.Box)
			}
		}
	} else {
		fmt.Println("ℹ️  No test image found. Model is ready for use!")
		fmt.Println("\n📋 Usage Examples:")
		fmt.Println("==================")
		fmt.Println("1. Run with video file:")
		fmt.Println("   go run main.go --video fall5.mp4 --fastrcnn --onnx-model fasterrcnn.onnx")
		fmt.Println()
		fmt.Println("2. Run with camera:")
		fmt.Println("   go run main.go --fastrcnn --onnx-model fasterrcnn.onnx")
		fmt.Println()
		fmt.Println("3. Test mode (motion detection only):")
		fmt.Println("   go run main.go --video fall5.mp4 --test-mode")
	}

	fmt.Println("\n🎉 FastRCNN model test completed!")
} 
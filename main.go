package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nvr-ai/go-ml/motion"
	"github.com/nvr-ai/go-ml/onnx"
	"github.com/nvr-ai/go-ml/profiler"
	"gocv.io/x/gocv"
)

const (
	// window is a flag to indicate if a window should be created to display the video.
	showWindow = false
	// deviceID is the ID of the video capture device to use.
	deviceID = 0
	// MinimumArea represents the minimum area threshold for motion detection.
	MinimumArea = 300000
	// DefaultMinMotionDuration is the default minimum duration for motion events.
	DefaultMinMotionDuration = 1500 * time.Millisecond
	// Default ONNX model path
	DefaultONNXModelPath = "yolov3u.onnx"
	// Default output directory for saved frames
	DefaultOutputDir = "motion_frames"
)

// Supported file extensions
var (
	supportedVideoExtensions = []string{".mp4", ".avi", ".mov"}
	supportedImageExtensions = []string{".jpg", ".jpeg", ".png", ".bmp"}
)

// InputType represents the type of input being processed
type InputType int

const (
	InputCamera InputType = iota
	InputVideo
	InputImage
)

// InputConfig holds the input configuration
type InputConfig struct {
	Type     InputType
	Path     string
	DeviceID int
}

func main() {
	// Parse command line arguments
	var (
		minDuration           time.Duration
		onnxModelPath         string
		confidenceThreshold   float64
		outputDir             string
		enableObjectDetection bool
		videoPath             string
		imagePath             string
		showVisualization     bool
	)
	flag.DurationVar(&minDuration, "min-duration", DefaultMinMotionDuration, "Minimum motion duration before reporting")
	flag.StringVar(&onnxModelPath, "onnx-model", DefaultONNXModelPath, "Path to YOLO ONNX model file")
	flag.Float64Var(&confidenceThreshold, "confidence", 0.5, "Object detection confidence threshold")
	flag.StringVar(&outputDir, "output-dir", DefaultOutputDir, "Output directory for saved frames")
	flag.BoolVar(&enableObjectDetection, "object-detection", true, "Enable object detection verification")
	flag.StringVar(&videoPath, "video", "", "Path to video file (.mp4, .avi, .mov)")
	flag.StringVar(&imagePath, "image", "", "Path to image file (.jpg, .jpeg, .png, .bmp)")
	flag.BoolVar(&showVisualization, "show-window", false, "Show visualization window")
	flag.Parse()

	// Validate input flags
	inputConfig, err := validateInputFlags(videoPath, imagePath)
	if err != nil {
		log.Fatal(err)
	}

	// Initialize object detection if enabled
	var objectDetector *onnx.ONNXDetector
	if enableObjectDetection {
		var err error
		objectDetector, err = onnx.NewSession(onnx.Config{
			ModelPath:           onnxModelPath,
			InputShape:          image.Point{X: 416, Y: 416},
			ConfidenceThreshold: float32(confidenceThreshold),
			NMSThreshold:        0.5,
			RelevantClasses:     []string{"person", "car", "truck", "bus", "motorcycle", "bicycle"},
		})
		if err != nil {
			fmt.Printf("⚠️  Warning: Failed to initialize object detector: %v\n", err)
			fmt.Printf("💡 This could be due to:\n")
			fmt.Printf("   - Incompatible ONNX model format\n")
			fmt.Printf("   - Missing OpenCV DNN support\n")
			fmt.Printf("   - Corrupted model file\n")
			fmt.Printf("   - Insufficient system resources\n")
			fmt.Printf("🔄 Continuing with motion detection only...\n")
			enableObjectDetection = false
		} else {
			defer objectDetector.Close()
			fmt.Printf("✅ Object detector initialized successfully\n")
			fmt.Printf("Model loaded successfully: yolov3u.onnx\n")
		}
	}

	// Create output directory if it doesn't exist
	if enableObjectDetection {
		if err := os.MkdirAll(outputDir, 0o755); err != nil {
			fmt.Printf("Warning: Failed to create output directory: %v\n", err)
		}
	}

	// Initialize video capture based on input type
	var webcam *gocv.VideoCapture

	switch inputConfig.Type {
	case InputCamera:
		var err error
		webcam, err = gocv.OpenVideoCapture(inputConfig.DeviceID)
		if err != nil {
			log.Fatalf("Error opening video capture device: %v", inputConfig.DeviceID)
		}
		fmt.Printf("Starting motion detection on camera device: %v\n", inputConfig.DeviceID)
	case InputVideo:
		var err error
		webcam, err = gocv.OpenVideoCapture(inputConfig.Path)
		if err != nil {
			log.Fatalf("Error opening video file: %v", inputConfig.Path)
		}
		fmt.Printf("Processing video: %s\n", inputConfig.Path)
	case InputImage:
		fmt.Printf("Processing image: %s\n", inputConfig.Path)
		// For image processing, we'll handle it differently
		processImage(inputConfig.Path, objectDetector, enableObjectDetection, outputDir, onnxModelPath, confidenceThreshold)
		return
	}
	defer webcam.Close()

	// Initialize a Mat to store the current frame.
	img := gocv.NewMat()
	defer img.Close()

	// Used to store the difference between the current frame and the background.
	imgDelta := gocv.NewMat()
	defer imgDelta.Close()

	// Thresholded image.
	imgThresh := gocv.NewMat()
	defer imgThresh.Close()

	// Initialize the background subtractor.
	// This is used to subtract the background from the current frame.
	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer mog2.Close()

	// Create the motion detector instance with the given configuration.
	detector := motion.New(motion.Config{
		MinMotionDuration: minDuration,
		MinimumArea:       MinimumArea,
	})
	defer detector.Close()

	fmt.Printf("\n🚀 Motion Detection System Started\n")
	fmt.Printf("=====================================\n")
	fmt.Printf("⚙️  Configuration:\n")
	fmt.Printf("   📅 Minimum motion duration: %v\n", minDuration)
	fmt.Printf("   📏 Minimum motion area: %d\n", MinimumArea)
	fmt.Printf("   🎥 Input type: %s\n", func() string {
		switch inputConfig.Type {
		case InputCamera:
			return fmt.Sprintf("Camera (Device %d)", inputConfig.DeviceID)
		case InputVideo:
			return fmt.Sprintf("Video: %s", inputConfig.Path)
		case InputImage:
			return fmt.Sprintf("Image: %s", inputConfig.Path)
		default:
			return "Unknown"
		}
	}())
	if enableObjectDetection {
		fmt.Printf("   🤖 Object detection: ✅ Enabled\n")
		fmt.Printf("   🎯 Model: %s\n", onnxModelPath)
		fmt.Printf("   📊 Confidence threshold: %.2f\n", confidenceThreshold)
		fmt.Printf("   💾 Output directory: %s\n", outputDir)
		fmt.Printf("   🎯 Relevant classes: person, car, truck, bus, motorcycle, bicycle\n")
	} else {
		fmt.Printf("   🤖 Object detection: ❌ Disabled\n")
	}
	fmt.Printf("   📈 Profiling: ✅ Enabled\n")
	fmt.Printf("   🖼️  Show window: %t\n", showVisualization)
	fmt.Printf("   ⚙️  Tuning parameters:\n")
	fmt.Printf("      - Motion area threshold: %d pixels\n", MinimumArea)
	fmt.Printf("      - Motion duration: %v\n", minDuration)
	fmt.Printf("      - Confidence threshold: %.2f\n", confidenceThreshold)
	fmt.Printf("      - NMS threshold: 0.50\n")
	fmt.Printf("=====================================\n\n")

	profiler := profiler.NewRuntimeProfiler(profiler.ProfilingOptions{
		ReportInterval: 2 * time.Second,
		SampleInterval: 100 * time.Millisecond,
		MaxSamples:     600,
	})

	profiler.Start()

	var window *gocv.Window
	if showVisualization {
		// Create a window to display the video.
		window = gocv.NewWindow("Motion Detection")
		defer window.Close()
		fmt.Printf("🖼️  Visualization window enabled\n")
	}

	frameCounter := 0
	for {
		// Time the frame processing.
		frameStart := time.Now()
		stopTiming := profiler.StartOperation("frame_processing")

		// Read the next frame from the video capture device.
		if ok := webcam.Read(&img); !ok {
			if inputConfig.Type == InputVideo {
				fmt.Printf("End of video file: %v\n", inputConfig.Path)
			} else {
				fmt.Printf("Device closed: %v\n", inputConfig.DeviceID)
			}
			return
		}
		if img.Empty() {
			stopTiming()
			continue
		}

		// Apply background subtraction to get foreground out of the current frame.
		mog2.Apply(img, &imgDelta)

		// Threshold to get binary image so that we can find contours.
		gocv.Threshold(imgDelta, &imgThresh, 25, 255, gocv.ThresholdBinary)

		// Dilate to fill gaps in the binary image.
		gocv.Dilate(imgThresh, &imgThresh, detector.Kernel)

		// Find contours by using the external retrieval method and the simple approximation method.
		contours := gocv.FindContours(imgThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		defer contours.Close()

		var (
			motionDetected    bool
			maxArea           float64
			largestContourIdx int
			motionROIs        []image.Rectangle
		)

		for i := 0; i < contours.Size(); i++ {
			area := gocv.ContourArea(contours.At(i))
			if area >= detector.MinimumArea {
				motionDetected = true
				if area > maxArea {
					maxArea = area
					largestContourIdx = i
				}
				// Store motion ROI for object detection
				rect := gocv.BoundingRect(contours.At(i))
				motionROIs = append(motionROIs, rect)
			}
		}

		// Process motion state.
		reportMotion, status := detector.Process(motionDetected, maxArea)

		// Update FPS.
		detector.FPS(motionDetected)

		// Record frame processing time.
		detector.FrameProcessingTime = time.Since(frameStart)
		stopTiming()

		// Object detection on motion ROIs
		var relevantObjectsDetected bool
		var detectedObjects []string
		if enableObjectDetection && motionDetected && len(motionROIs) > 0 {
			relevantObjectsDetected, detectedObjects = processMotionROIs(img, motionROIs, objectDetector, frameCounter, outputDir)
		}

		// Draw contours and bounding boxes for significant motion.
		if motionDetected && largestContourIdx >= 0 {
			// Draw all significant contours.
			for i := 0; i < contours.Size(); i++ {
				if gocv.ContourArea(contours.At(i)) >= detector.MinimumArea {
					gocv.DrawContours(&img, contours, i, color.RGBA{0, 0, 255, 0}, 2)
					rect := gocv.BoundingRect(contours.At(i))
					gocv.Rectangle(&img, rect, color.RGBA{0, 0, 255, 0}, 2)
				}
			}
		}

		// Report motion event if duration threshold met
		if reportMotion {
			if enableObjectDetection && relevantObjectsDetected {
				fmt.Printf("[%s] 🎯 Motion event with relevant objects detected - Area: %.0f, Events: %d\n",
					time.Now().Format("15:04:05.000"), maxArea, detector.MotionEventCount)
			} else if enableObjectDetection {
				fmt.Printf("[%s] ⚠️  Motion event detected (no relevant objects) - Area: %.0f, Events: %d\n",
					time.Now().Format("15:04:05.000"), maxArea, detector.MotionEventCount)
			} else {
				fmt.Printf("[%s] 🔍 Motion event detected - Area: %.0f, Events: %d\n",
					time.Now().Format("15:04:05.000"), maxArea, detector.MotionEventCount)
			}
		}

		// Print performance metrics for each frame with proper type handling
		processingTimeMs := float64(detector.FrameProcessingTime.Microseconds()) / 1000.0
		fmt.Printf("[Frame %d] FPS: %.1f | Motion FPS: %.1f | Processing: %.2fms | Motion: %t | Events: %d",
			frameCounter, detector.CurrentFPS, detector.MotionFPS, processingTimeMs,
			motionDetected, detector.MotionEventCount)

		if motionDetected {
			fmt.Printf(" | Area: %.0f | ROIs: %d", maxArea, len(motionROIs))
		}

		if enableObjectDetection {
			if relevantObjectsDetected && len(detectedObjects) > 0 {
				fmt.Printf(" | Objects: ✅ Found (%s)", strings.Join(detectedObjects, ", "))
			} else {
				fmt.Printf(" | Objects: ❌ None")
			}
		}

		fmt.Printf("\n")

		// Draw status information
		gocv.PutText(&img, status, image.Pt(10, 30), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 0, 255, 0}, 2)

		// Draw FPS information
		fpsText := fmt.Sprintf("FPS: %.1f | Motion FPS: %.1f", detector.CurrentFPS, detector.MotionFPS)
		gocv.PutText(&img, fpsText, image.Pt(10, 60), gocv.FontHersheyPlain, 1.2, color.RGBA{255, 255, 255, 0}, 2)

		// Draw motion event count
		eventText := fmt.Sprintf("Motion Events: %d", detector.MotionEventCount)
		gocv.PutText(&img, eventText, image.Pt(10, 90), gocv.FontHersheyPlain, 1.2, color.RGBA{255, 255, 255, 0}, 2)

		// Draw object detection status
		if enableObjectDetection {
			objText := fmt.Sprintf("Object Detection: %s", func() string {
				if relevantObjectsDetected {
					return "Relevant Objects Found"
				}
				return "No Relevant Objects"
			}())
			gocv.PutText(&img, objText, image.Pt(10, 120), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 255, 0, 0}, 2)
		}

		// Show the image
		if showVisualization {
			window.IMShow(img)
		}

		frameCounter++
	}
}

// validateInputFlags validates the input flags and returns the input configuration
func validateInputFlags(videoPath, imagePath string) (*InputConfig, error) {
	// Check if both or neither are provided
	if videoPath != "" && imagePath != "" {
		return nil, fmt.Errorf("error: cannot specify both --video and --image flags")
	}
	if videoPath == "" && imagePath == "" {
		// Default to camera
		return &InputConfig{Type: InputCamera, DeviceID: deviceID}, nil
	}

	// Validate video input
	if videoPath != "" {
		if err := validateFile(videoPath, supportedVideoExtensions); err != nil {
			return nil, fmt.Errorf("video validation error: %w", err)
		}
		return &InputConfig{Type: InputVideo, Path: videoPath}, nil
	}

	// Validate image input
	if imagePath != "" {
		if err := validateFile(imagePath, supportedImageExtensions); err != nil {
			return nil, fmt.Errorf("image validation error: %w", err)
		}
		return &InputConfig{Type: InputImage, Path: imagePath}, nil
	}

	return nil, fmt.Errorf("unexpected input configuration")
}

// validateFile checks if the file exists and has a supported extension
func validateFile(filePath string, supportedExtensions []string) error {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return fmt.Errorf("file not found: %s", filePath)
	}

	// Check file extension
	ext := strings.ToLower(filepath.Ext(filePath))
	for _, supportedExt := range supportedExtensions {
		if ext == supportedExt {
			return nil
		}
	}

	return fmt.Errorf("unsupported file extension: %s. Supported extensions: %v", ext, supportedExtensions)
}

// processImage processes a single image file
func processImage(imagePath string, objectDetector *onnx.ONNXDetector, enableObjectDetection bool, outputDir, onnxModelPath string, confidenceThreshold float64) {
	// Load the image
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		log.Fatalf("Error reading image: %s", imagePath)
	}
	defer img.Close()

	fmt.Printf("Processing image: %s\n", imagePath)
	fmt.Printf("Image size: %dx%d\n", img.Cols(), img.Rows())

	if enableObjectDetection && objectDetector != nil {
		// Run object detection on the entire image
		detections, err := objectDetector.Detect(img)
		if err != nil {
			fmt.Printf("Error detecting objects: %v\n", err)
		} else {
			fmt.Printf("Found %d objects\n", len(detections))

			// Process detections
			for i, detection := range detections {
				if objectDetector.IsRelevantClass(detection.ClassName) {
					fmt.Printf("Object %d: %s (confidence: %.2f) at %v\n",
						i+1, detection.ClassName, detection.Score, detection.Box)

					// Draw detection box on the image
					gocv.Rectangle(&img, detection.Box, color.RGBA{0, 255, 0, 0}, 2)
					label := fmt.Sprintf("%s %.2f", detection.ClassName, detection.Score)
					gocv.PutText(&img, label, detection.Box.Min, gocv.FontHersheyPlain, 0.8, color.RGBA{0, 255, 0, 0}, 2)
				}
			}
		}
	}

	// Save the processed image
	outputPath := filepath.Join(outputDir, "processed_"+filepath.Base(imagePath))
	if gocv.IMWrite(outputPath, img) {
		fmt.Printf("Processed image saved to: %s\n", outputPath)
	} else {
		fmt.Printf("Failed to save processed image\n")
	}
}

// processMotionROIs runs object detection on motion regions of interest
func processMotionROIs(img gocv.Mat, motionROIs []image.Rectangle, detector *onnx.ONNXDetector, frameCounter int, outputDir string) (bool, []string) {
	var relevantObjectsDetected bool
	var detectedObjects []string
	objectCount := make(map[string]int)

	for _, roi := range motionROIs {
		// Run object detection on the ROI
		detections, err := detector.DetectROI(img, roi)
		if err != nil {
			fmt.Printf("Error detecting objects in ROI: %v\n", err)
			continue
		}

		// Check if any relevant objects were detected
		for _, detection := range detections {
			if detector.IsRelevantClass(detection.ClassName) {
				relevantObjectsDetected = true
				objectCount[detection.ClassName]++

				// Save the frame with relevant objects
				filename := filepath.Join(outputDir, fmt.Sprintf("motion_frame_%d_%s_%.2f.jpg",
					frameCounter, detection.ClassName, detection.Score))

				if gocv.IMWrite(filename, img) {
					fmt.Printf("💾 Saved frame with %s (confidence: %.2f) to %s\n",
						detection.ClassName, detection.Score, filename)
				} else {
					fmt.Printf("❌ Failed to save frame to %s\n", filename)
				}

				// Draw detection box on the image
				gocv.Rectangle(&img, detection.Box, color.RGBA{0, 255, 0, 0}, 2)
				label := fmt.Sprintf("%s %.2f", detection.ClassName, detection.Score)
				gocv.PutText(&img, label, detection.Box.Min, gocv.FontHersheyPlain, 0.8, color.RGBA{0, 255, 0, 0}, 2)
			}
		}
	}

	// Convert object count to string slice
	for objType, count := range objectCount {
		if count > 0 {
			detectedObjects = append(detectedObjects, fmt.Sprintf("%s(%d)", objType, count))
		}
	}

	return relevantObjectsDetected, detectedObjects
}

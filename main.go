package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"time"

	"github.com/nvr-ai/go-ml/motion"
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
)

func main() {
	// Parse command line arguments
	var minDuration time.Duration
	flag.DurationVar(&minDuration, "min-duration", DefaultMinMotionDuration, "Minimum motion duration before reporting")
	flag.Parse()

	args := flag.Args()
	if len(args) < 1 {
		fmt.Println("How to run:\n\tmotion-detect [camera ID] [--min-duration=500ms]")
		return
	}

	// Initialize video capture.
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
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

	fmt.Printf("Starting motion detection on device: %v\n", deviceID)
	fmt.Printf("Minimum motion duration: %v\n", minDuration)

	profiler := profiler.NewRuntimeProfiler(profiler.ProfilingOptions{
		ReportInterval: 2 * time.Second,
		SampleInterval: 100 * time.Millisecond,
		MaxSamples:     600,
	})

	profiler.Start()

	var window *gocv.Window
	if showWindow {
		// Create a window to display the video.
		window := gocv.NewWindow("Motion Detection")
		defer window.Close()
	}

	for {
		// Time the frame processing.
		frameStart := time.Now()
		stopTiming := profiler.StartOperation("frame_processing")

		// Read the next frame from the video capture device.
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
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
		)

		for i := 0; i < contours.Size(); i++ {
			area := gocv.ContourArea(contours.At(i))
			if area >= detector.MinimumArea {
				motionDetected = true
				if area > maxArea {
					maxArea = area
					largestContourIdx = i
				}
			}
		}

		// Process motion state.
		reportMotion, status := detector.Process(motionDetected, maxArea)

		// Update FPS.
		detector.FPS(motionDetected)

		// Record frame processing time.
		detector.FrameProcessingTime = time.Since(frameStart)
		stopTiming()

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
			fmt.Printf("[%s] Motion event detected - Area: %.0f, Events: %d\n",
				time.Now().Format("15:04:05.000"), maxArea, detector.MotionEventCount)
		}

		// Draw status information
		gocv.PutText(&img, status, image.Pt(10, 30), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 0, 255, 0}, 2)

		// Draw FPS information
		fpsText := fmt.Sprintf("FPS: %.1f | Motion FPS: %.1f", detector.CurrentFPS, detector.MotionFPS)
		gocv.PutText(&img, fpsText, image.Pt(10, 60), gocv.FontHersheyPlain, 1.2, color.RGBA{255, 255, 255, 0}, 2)

		// Draw motion event count
		eventText := fmt.Sprintf("Motion Events: %d", detector.MotionEventCount)
		gocv.PutText(&img, eventText, image.Pt(10, 90), gocv.FontHersheyPlain, 1.2, color.RGBA{255, 255, 255, 0}, 2)

		// Show the image
		if showWindow {
			window.IMShow(img)
		}
	}
}

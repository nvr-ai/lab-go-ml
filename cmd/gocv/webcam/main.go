package main

import (
	"fmt"
	"image/color"
	"time"

	"gocv.io/x/gocv"
)

func main() {
	// set to use a video capture device 0
	deviceID := 0

	// open webcam
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer webcam.Close()

	// open display window
	window := gocv.NewWindow("Face Detect")
	defer window.Close()

	// prepare image matrix
	img := gocv.NewMat()
	defer img.Close()

	// color for the rect when faces detected
	blue := color.RGBA{0, 0, 255, 0}

	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load("haarcascade_frontalface_default.xml") {
		fmt.Println("Error reading cascade file: haarcascade_frontalface_default.xml")
		return
	}

	// FPS tracking variables
	fps := 0.0
	frameCount := 0
	lastTime := time.Now()

	fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("cannot read device %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// Update FPS calculation
		frameCount++
		currentTime := time.Now()
		elapsed := currentTime.Sub(lastTime).Seconds()

		// Calculate FPS every second
		if elapsed >= 1.0 {
			fps = float64(frameCount) / elapsed
			frameCount = 0
			lastTime = currentTime
		}

		// detect faces
		rects := classifier.DetectMultiScale(img)
		fmt.Printf("found %d faces | FPS: %.2f\n", len(rects), fps)

		// draw a rectangle around each face on the original image
		for _, r := range rects {
			gocv.Rectangle(&img, r, blue, 3)
		}

		// show the image in the window, and wait 1 millisecond
		window.IMShow(img)
		window.WaitKey(1)
	}
}

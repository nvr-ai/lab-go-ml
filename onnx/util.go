package onnx

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"runtime"
	"sort"

	"github.com/nfnt/resize"
	"github.com/nvr-ai/go-ml/common"
	ort "github.com/yalue/onnxruntime_go"
)

// COCO Classes for YOLO models
var COCOClasses = []string{
	"__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
	"frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
	"bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// Array of YOLOv8 class labels
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// GetCOCOClasses returns the COCO class names
func GetCOCOClasses() []string {
	return COCOClasses
}

// GetClassMapping returns a mapping of class names to their IDs
func GetClassMapping() map[string]int {
	mapping := make(map[string]int)
	for i, className := range COCOClasses {
		mapping[className] = i
	}
	return mapping
}

// Helper functions
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/libonnxruntime.1.23.0.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "./third_party/libonnxruntime.1.23.0.dylib"
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

// DecodeJPEGBytes decodes a JPEG image from a byte slice and returns an image.Image.
//
// Arguments:
//   - data: The JPEG image data as a byte slice.
//
// Returns:
//   - image.Image: The decoded image.
//   - error: An error if decoding fails.
func DecodeJPEGBytes(data []byte) (image.Image, error) {
	reader := bytes.NewReader(data)
	img, err := jpeg.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode JPEG: %w", err)
	}
	return img, nil
}

// DecodeJPEG decodes a JPEG byte slice into an image.Image with minimal overhead.
// It avoids unnecessary allocations and uses a zero-copy reader.
func DecodeJPEG(data []byte) (image.Image, error) {
	return jpeg.Decode(bytes.NewReader(data))
}

// PrepareInput prepares the input for the ONNX model.
//
// Arguments:
//   - pic: The image to prepare.
//   - dst: The destination tensor to populate.
//
// Returns:
//   - error: An error if the input preparation fails.
func PrepareInput(img image.Image, dst *ort.Tensor[float32]) error {
	data := dst.GetData()
	channelSize := 640 * 640
	if len(data) < (channelSize * 3) {
		return fmt.Errorf("Destination tensor only holds %d floats, needs "+
			"%d (make sure it's the right shape!)", len(data), channelSize*3)
	}
	red := data[0:channelSize]
	green := data[channelSize : channelSize*2]
	blue := data[channelSize*2 : channelSize*3]

	// Resize the image to 640x640 using Lanczos3 algorithm.
	img = resize.Resize(640, 640, img, resize.Lanczos3)

	i := 0
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			red[i] = float32(r>>8) / 255.0
			green[i] = float32(g>>8) / 255.0
			blue[i] = float32(b>>8) / 255.0
			i++
		}
	}
	return nil
}

func ProcessInferenceOutput(output []float32, originalWidth,
	originalHeight int,
) []common.BoundingBox {
	boundingBoxes := make([]common.BoundingBox, 0, 8400)

	var classID int
	var probability float32

	// Iterate through the output array, considering 8400 indices
	for idx := 0; idx < 8400; idx++ {
		// Iterate through 80 classes and find the class with the highest probability
		probability = -1e9
		for col := 0; col < 80; col++ {
			currentProb := output[8400*(col+4)+idx]
			if currentProb > probability {
				probability = currentProb
				classID = col
			}
		}

		// If the probability is less than 0.5, continue to the next index
		if probability < 0.5 {
			continue
		}

		// Extract the coordinates and dimensions of the bounding box
		xc, yc := output[idx], output[8400+idx]
		w, h := output[2*8400+idx], output[3*8400+idx]
		x1 := (xc - w/2) / 640 * float32(originalWidth)
		y1 := (yc - h/2) / 640 * float32(originalHeight)
		x2 := (xc + w/2) / 640 * float32(originalWidth)
		y2 := (yc + h/2) / 640 * float32(originalHeight)

		// Append the bounding box to the result
		boundingBoxes = append(boundingBoxes, common.BoundingBox{
			Label:      yoloClasses[classID],
			Confidence: probability,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		})
	}

	// Sort the bounding boxes by probability
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].Confidence < boundingBoxes[j].Confidence
	})

	// Define a slice to hold the final result
	mergedResults := make([]common.BoundingBox, 0, len(boundingBoxes))

	// Iterate through sorted bounding boxes, removing overlaps
	for _, candidateBox := range boundingBoxes {
		overlapsExistingBox := false
		for _, existingBox := range mergedResults {
			if candidateBox.IOU(&existingBox) > 0.7 {
				overlapsExistingBox = true
				break
			}
		}
		if !overlapsExistingBox {
			mergedResults = append(mergedResults, candidateBox)
		}
	}

	// This will still be in sorted order by confidence
	return mergedResults
}

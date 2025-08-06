package detectors

import (
	"sort"

	"github.com/nvr-ai/go-ml/common"
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models"
)

// InferenceResult holds a single detection’s output.
type Result struct {
	Class models.OutputClass
	Score float64
	Box   images.Rect
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
			Label:      models.YOLOClasses.Classes[classID].Name,
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

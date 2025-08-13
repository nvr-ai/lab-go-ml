// Package yolov4 - postprocess YOLOv4 model outputs.
package yolov4

import (
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// PostProcess postprocesses the output of the YOLOv4 model.
//
// Arguments:
//   - output: The output of the YOLOv4 model.
//   - cfg: The configuration for the YOLOv4 model.
//
// Returns:
//   - A slice of postprocessed results.
func (p *YOLOv4) PostProcess(output []float32, config *postprocess.NMSConfig) []postprocess.Result {
	const numCols = 85
	if len(output)%numCols != 0 {
		return nil
	}
	numRows := len(output) / numCols
	results := make([]postprocess.Result, 0, numRows)

	for i := 0; i < numRows; i++ {
		offset := i * numCols
		objConf := output[offset+4]
		if objConf < config.IoUThreshold {
			continue
		}

		classID := 0
		maxScore := float32(0)
		for j := 5; j < numCols; j++ {
			score := output[offset+j]
			if score > maxScore {
				maxScore = score
				classID = j - 5
			}
		}

		finalScore := objConf * maxScore
		if finalScore < config.IoUThreshold {
			continue
		}

		w := output[offset+2]
		h := output[offset+3]

		results = append(results, postprocess.Result{
			Box: images.Rect{
				X1: output[offset+0] - w/2, // cx - w/2
				Y1: output[offset+1] - h/2, // cy - h/2
				X2: output[offset+0] + w/2, // cx + w/2
				Y2: output[offset+1] + h/2, // cy + h/2
			},
			Score: finalScore,
			Class: classID,
		})
	}

	if config.Greedy {
		return postprocess.ApplyGreedyNMS(results, config)
	}
	return postprocess.ApplyNMS(results, config)
}

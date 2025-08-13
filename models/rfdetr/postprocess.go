// Package rfdetr - postprocess RF-DETR model outputs.
package rfdetr

import (
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// PostProcess postprocesses the output of the RF-DETR model.
//
// Arguments:
//   - output: The output of the RF-DETR model.
//   - cfg: The configuration for the RF-DETR model.
//
// Returns:
//   - A slice of postprocessed results.
func (p *RFDETR) PostProcess(output []float32, config *postprocess.NMSConfig) []postprocess.Result {
	const rowSize = 6
	if len(output)%rowSize != 0 {
		return nil // Malformed output
	}

	numRows := len(output) / rowSize
	results := make([]postprocess.Result, 0, numRows)

	for i := 0; i < numRows; i++ {
		offset := i * rowSize
		score := output[offset+4]
		if score < config.IoUThreshold {
			continue
		}
		results = append(results, postprocess.Result{
			Box: images.Rect{
				X1: output[offset+0],
				Y1: output[offset+1],
				X2: output[offset+2],
				Y2: output[offset+3],
			},
			Score: score,
			Class: int(output[offset+5]),
		})
	}

	if config.Greedy {
		return postprocess.ApplyGreedyNMS(results, config)
	}
	return postprocess.ApplyNMS(results, config)
}

// Package dfine - postprocess D-FINE model outputs.
package dfine

import (
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// PostProcess transforms the output of the D-FINE model inference into a
// slice of postprocessed results by:
//   - Filtering out results with a confidence score below the IoU threshold.
//   - Applying Greedy NMS if the Greedy flag is set.
//   - Applying NMS if the Greedy flag is not set.
//   - Returning the postprocessed results.
//
// Arguments:
//   - output: The output of the D-FINE model.
//   - cfg: The configuration for the D-FINE model.
//
// Returns:
//   - A slice of postprocessed results.
func (p *DFINE) PostProcess(output []float32, config *postprocess.NMSConfig) []postprocess.Result {
	const rowSize = 6
	if len(output)%rowSize != 0 {
		return nil
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

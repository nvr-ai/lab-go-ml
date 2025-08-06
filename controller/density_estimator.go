// Package controller - This file contains the density estimator for estimating the density of detections.
package controller

// DensityEstimator is an interface for a density estimator.
type DensityEstimator interface {
	EstimateDensity(detections []Detection) (int, error)
}

// StandardDensityEstimator is a density estimator.
type StandardDensityEstimator struct {
	MinBoxArea int
}

// EstimateDensity estimates the density of detections.
//
// Arguments:
//   - detections: The detections to estimate the density of.
//
// Returns:
//   - int: The density of the detections.
//   - error: An error if the estimation fails.
func (a *StandardDensityEstimator) EstimateDensity(detections []Detection) (int, error) {
	count := 0
	for _, d := range detections {
		area := (d.BBox.Max.X - d.BBox.Min.X) * (d.BBox.Max.Y - d.BBox.Min.Y)
		if area < a.MinBoxArea {
			count++ // small object
		}
	}
	return count, nil
}

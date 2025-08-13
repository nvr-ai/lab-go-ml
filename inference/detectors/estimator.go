// Package detectors - Advanced density estimation for object detection analysis
package detectors

import (
	"fmt"
	"image"
	"math"
	"sort"
	"sync"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/models/postprocess"
)

// DensityEstimator is an interface for estimating object density in frames.
//
// Implementations should analyze the spatial distribution and characteristics
// of detected objects to determine scene complexity and density.
type DensityEstimator interface {
	EstimateDensity(detections []postprocess.Result) (int, error)
	GetDensityMetrics(detections []postprocess.Result) (*DensityMetrics, error)
}

// DensityMetrics provides detailed analysis of object density and distribution.
//
// This structure contains comprehensive metrics about detected objects including
// spatial distribution, size analysis, and clustering information.
type DensityMetrics struct {
	// TotalObjects is the total number of detected objects
	TotalObjects int `json:"total_objects"`

	// SmallObjects is the count of objects below the small object threshold
	SmallObjects int `json:"small_objects"`

	// LargeObjects is the count of objects above the large object threshold
	LargeObjects int `json:"large_objects"`

	// AverageObjectSize is the mean bounding box area of all objects
	AverageObjectSize float64 `json:"average_object_size"`

	// ObjectSizeVariance measures the spread in object sizes
	ObjectSizeVariance float64 `json:"object_size_variance"`

	// SpatialDensity measures objects per unit area (objects per 1000 pixels)
	SpatialDensity float64 `json:"spatial_density"`

	// ClusteringCoefficient measures how clustered objects are (0-1 scale)
	ClusteringCoefficient float64 `json:"clustering_coefficient"`

	// OverlapRatio is the fraction of objects that overlap with others
	OverlapRatio float64 `json:"overlap_ratio"`

	// CenterOfMass represents the center point of all detections
	CenterOfMass image.Point `json:"center_of_mass"`

	// BoundingRegion contains all detections
	BoundingRegion image.Rectangle `json:"bounding_region"`

	// ConfidenceDistribution provides statistics on detection confidence
	ConfidenceDistribution ConfidenceStats `json:"confidence_distribution"`
}

// ConfidenceStats provides statistical analysis of detection confidence scores.
type ConfidenceStats struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	StdDev float64 `json:"std_dev"`
}

// DensityEstimationConfig contains parameters for density estimation algorithms.
type DensityEstimationConfig struct {
	// MinBoxArea defines the minimum area for an object to be considered significant
	MinBoxArea int `json:"min_box_area"`

	// LargeObjectThreshold defines the area above which objects are considered large
	LargeObjectThreshold int `json:"large_object_threshold"`

	// ClusteringRadius defines the distance threshold for clustering analysis
	ClusteringRadius float64 `json:"clustering_radius"`

	// OverlapThreshold defines the IoU threshold for overlap detection
	OverlapThreshold float64 `json:"overlap_threshold"`

	// FrameArea represents the total frame area for spatial density calculations
	FrameArea int `json:"frame_area"`

	// EnableAdvancedMetrics toggles computation of expensive metrics
	EnableAdvancedMetrics bool `json:"enable_advanced_metrics"`

	// WeightedByConfidence applies confidence-based weighting to density calculations
	WeightedByConfidence bool `json:"weighted_by_confidence"`
}

// DefaultDensityEstimationConfig returns a default configuration for density estimation
func DefaultDensityEstimationConfig() DensityEstimationConfig {
	return DensityEstimationConfig{
		MinBoxArea:            500,
		LargeObjectThreshold:  5000,
		ClusteringRadius:      100.0,
		OverlapThreshold:      0.3,
		FrameArea:             640 * 640, // Default for 640x640 images
		EnableAdvancedMetrics: true,
		WeightedByConfidence:  true,
	}
}

// DefaultDensityEstimator provides sophisticated object density analysis.
//
// This estimator analyzes spatial distribution, object clustering, size distribution,
// and confidence patterns to provide comprehensive density metrics suitable for
// dynamic resolution control and scene complexity assessment.
type DefaultDensityEstimator struct {
	config DensityEstimationConfig
	mu     sync.RWMutex
}

// NewDensityEstimator creates a new advanced density estimator.
//
// Arguments:
//   - config: Configuration parameters for density estimation
//
// Returns:
//   - *AdvancedDensityEstimator: The initialized density estimator
//
// @example
// config := DefaultDensityEstimationConfig()
// estimator := NewDensityEstimator(config)
func NewDensityEstimator(config DensityEstimationConfig) *DefaultDensityEstimator {
	return &DefaultDensityEstimator{
		config: config,
	}
}

// EstimateDensity estimates the object density using multiple factors.
//
// This method provides a single density score suitable for resolution control.
// The score combines object count, spatial distribution, and confidence weighting
// to provide an optimal measure for dynamic resolution switching.
//
// Arguments:
//   - detections: The list of object detections to analyze
//
// Returns:
//   - int: Density score (typically 0-20+ for resolution control)
//   - error: An error if density estimation fails
//
// @example
// estimator := NewAdvancedDensityEstimator(DefaultDensityEstimationConfig())
// density, err := estimator.EstimateDensity(detections)
// fmt.Printf("Scene density: %d\n", density)
func (ade *DefaultDensityEstimator) EstimateDensity(detections []postprocess.Result) (int, error) {
	ade.mu.RLock()
	defer ade.mu.RUnlock()

	if len(detections) == 0 {
		return 0, nil
	}

	metrics, err := ade.calculateDensityMetrics(detections)
	if err != nil {
		return 0, fmt.Errorf("failed to calculate density metrics: %w", err)
	}

	// Multi-factor density score calculation
	var densityScore float64

	// Base object count (weighted by confidence if enabled)
	if ade.config.WeightedByConfidence {
		for _, detection := range detections {
			densityScore += float64(detection.Score)
		}
	} else {
		densityScore = float64(len(detections))
	}

	// Small object bonus (more complex scenes)
	smallObjectMultiplier := 1.5
	densityScore += float64(metrics.SmallObjects) * smallObjectMultiplier

	// Clustering penalty/bonus based on spatial distribution
	clusteringAdjustment := metrics.ClusteringCoefficient * 2.0
	densityScore += clusteringAdjustment

	// Overlap adjustment (higher overlap = more complex scene)
	overlapAdjustment := metrics.OverlapRatio * 3.0
	densityScore += overlapAdjustment

	// Spatial density contribution
	spatialContribution := metrics.SpatialDensity * 0.1
	densityScore += spatialContribution

	return int(math.Round(densityScore)), nil
}

// GetDensityMetrics provides comprehensive density analysis.
//
// This method calculates detailed metrics about object distribution, clustering,
// and characteristics that can be used for advanced scene analysis and optimization.
//
// Arguments:
//   - detections: The list of object detections to analyze
//
// Returns:
//   - *DensityMetrics: Comprehensive density analysis results
//   - error: An error if analysis fails
func (ade *DefaultDensityEstimator) GetDensityMetrics(
	detections []postprocess.Result,
) (*DensityMetrics, error) {
	ade.mu.RLock()
	defer ade.mu.RUnlock()

	return ade.calculateDensityMetrics(detections)
}

// calculateDensityMetrics performs the core density analysis calculations.
func (ade *DefaultDensityEstimator) calculateDensityMetrics(
	detections []postprocess.Result,
) (*DensityMetrics, error) {
	metrics := &DensityMetrics{
		TotalObjects: len(detections),
	}

	if len(detections) == 0 {
		return metrics, nil
	}

	// Calculate basic object statistics
	ade.calculateObjectSizeMetrics(detections, metrics)
	ade.calculateConfidenceMetrics(detections, metrics)
	ade.calculateSpatialMetrics(detections, metrics)

	// Calculate advanced metrics if enabled
	if ade.config.EnableAdvancedMetrics {
		ade.calculateClusteringMetrics(detections, metrics)
		ade.calculateOverlapMetrics(detections, metrics)
	}

	return metrics, nil
}

// calculateObjectSizeMetrics analyzes object size distribution
func (ade *DefaultDensityEstimator) calculateObjectSizeMetrics(
	detections []postprocess.Result,
	metrics *DensityMetrics,
) {
	var totalArea, sumSquaredDiff float64
	var areas []float64

	for _, detection := range detections {
		area := float64((detection.Box.X2 - detection.Box.X1) * (detection.Box.Y2 - detection.Box.Y1))
		areas = append(areas, area)
		totalArea += area

		// Count small and large objects
		if area < float64(ade.config.MinBoxArea) {
			metrics.SmallObjects++
		}
		if area > float64(ade.config.LargeObjectThreshold) {
			metrics.LargeObjects++
		}
	}

	metrics.AverageObjectSize = totalArea / float64(len(detections))

	// Calculate variance
	for _, area := range areas {
		diff := area - metrics.AverageObjectSize
		sumSquaredDiff += diff * diff
	}
	metrics.ObjectSizeVariance = sumSquaredDiff / float64(len(detections))
}

// calculateConfidenceMetrics analyzes detection confidence distribution
func (ade *DefaultDensityEstimator) calculateConfidenceMetrics(
	detections []postprocess.Result,
	metrics *DensityMetrics,
) {
	confidences := make([]float64, len(detections))
	var sum float64

	for i, detection := range detections {
		confidences[i] = float64(detection.Score)
		sum += float64(detection.Score)
	}

	sort.Float64s(confidences)

	stats := &metrics.ConfidenceDistribution
	stats.Mean = sum / float64(len(detections))
	stats.Min = confidences[0]
	stats.Max = confidences[len(confidences)-1]

	// Calculate median
	if len(confidences)%2 == 0 {
		stats.Median = (confidences[len(confidences)/2-1] + confidences[len(confidences)/2]) / 2
	} else {
		stats.Median = confidences[len(confidences)/2]
	}

	// Calculate standard deviation
	var sumSquaredDiff float64
	for _, conf := range confidences {
		diff := conf - stats.Mean
		sumSquaredDiff += diff * diff
	}
	stats.StdDev = math.Sqrt(sumSquaredDiff / float64(len(confidences)))
}

// calculateSpatialMetrics analyzes spatial distribution of objects.
func (ade *DefaultDensityEstimator) calculateSpatialMetrics(
	detections []postprocess.Result,
	metrics *DensityMetrics,
) {
	panic("not implemented")
	// var sumX, sumY float32
	// minX, minY := images.MaxFloat32(0, 0), images.MaxFloat32(0, 0)
	// maxX, maxY := images.MinFloat32(0, 0), images.MinFloat32(0, 0)

	// for _, detection := range detections {
	// 	centerX := (detection.Box.X1 + detection.Box.X2) / 2
	// 	centerY := (detection.Box.Y1 + detection.Box.Y2) / 2

	// 	sumX += centerX
	// 	sumY += centerY

	// 	// Update bounding region
	// 	if detection.Box.X1 < minX {
	// 		minX = detection.Box.X1
	// 	}
	// 	if detection.Box.X2 > maxX {
	// 		maxX = detection.Box.X2
	// 	}
	// 	if detection.Box.Y1 < minY {
	// 		minY = detection.Box.Y1
	// 	}
	// 	if detection.Box.Y2 > maxY {
	// 		maxY = detection.Box.Y2
	// 	}
	// }

	// metrics.CenterOfMass = image.Point{
	// 	X: sumX / len(detections),
	// 	Y: sumY / len(detections),
	// }

	// if minX != math.MaxInt32 {
	// 	metrics.BoundingRegion = image.Rect(minX, minY, maxX, maxY)
	// }

	// // Calculate spatial density (objects per 1000 pixels)
	// metrics.SpatialDensity = float64(len(detections)) / float64(ade.config.FrameArea) * 1000.0
}

// calculateClusteringMetrics analyzes how clustered objects are in space.
func (ade *DefaultDensityEstimator) calculateClusteringMetrics(
	detections []postprocess.Result,
	metrics *DensityMetrics,
) {
	if len(detections) < 2 {
		metrics.ClusteringCoefficient = 0.0
		return
	}

	totalPairs := 0
	clusteredPairs := 0

	for i := 0; i < len(detections); i++ {
		centerI := image.Point{
			X: int((detections[i].Box.X1 + detections[i].Box.X2) / 2),
			Y: int((detections[i].Box.Y1 + detections[i].Box.Y2) / 2),
		}

		for j := i + 1; j < len(detections); j++ {
			centerJ := image.Point{
				X: int((detections[j].Box.X1 + detections[j].Box.X2) / 2),
				Y: int((detections[j].Box.Y1 + detections[j].Box.Y2) / 2),
			}

			distance := math.Sqrt(
				math.Pow(float64(centerI.X-centerJ.X), 2) +
					math.Pow(float64(centerI.Y-centerJ.Y), 2),
			)

			totalPairs++
			if distance <= ade.config.ClusteringRadius {
				clusteredPairs++
			}
		}
	}

	if totalPairs > 0 {
		metrics.ClusteringCoefficient = float64(clusteredPairs) / float64(totalPairs)
	}
}

// calculateOverlapMetrics analyzes overlapping objects.
func (ade *DefaultDensityEstimator) calculateOverlapMetrics(
	detections []postprocess.Result,
	metrics *DensityMetrics,
) {
	if len(detections) < 2 {
		metrics.OverlapRatio = 0.0
		return
	}

	overlappingObjects := 0

	for i := 0; i < len(detections); i++ {
		hasOverlap := false

		for j := i + 1; j < len(detections); j++ {
			iou := images.CalculateIoU(
				detections[i].Box.X1,
				detections[i].Box.Y1,
				detections[i].Box.X2,
				detections[i].Box.Y2,
				detections[j].Box.X1,
				detections[j].Box.Y1,
				detections[j].Box.X2,
				detections[j].Box.Y2,
			)
			if iou >= float32(ade.config.OverlapThreshold) {
				hasOverlap = true
				break
			}
		}

		if hasOverlap {
			overlappingObjects++
		}
	}

	metrics.OverlapRatio = float64(overlappingObjects) / float64(len(detections))
}

// GetConfig returns the current density estimation configuration.
//
// Returns:
//   - DensityEstimationConfig: Current configuration parameters
func (ade *DefaultDensityEstimator) GetConfig() DensityEstimationConfig {
	ade.mu.RLock()
	defer ade.mu.RUnlock()
	return ade.config
}

// UpdateConfig updates the density estimation configuration.
//
// Arguments:
//   - config: New configuration parameters
func (ade *DefaultDensityEstimator) UpdateConfig(config DensityEstimationConfig) {
	ade.mu.Lock()
	defer ade.mu.Unlock()
	ade.config = config
}

// StandardDensityEstimator provides basic object counting for density estimation.
//
// This is a simplified implementation that maintains backward compatibility
// while providing basic density estimation functionality.
type StandardDensityEstimator struct {
	MinBoxArea int
}

// EstimateDensity provides simple object count-based density estimation.
//
// Arguments:
//   - detections: The detections to analyze
//
// Returns:
//   - int: Count of objects meeting the minimum size requirement
//   - error: An error if estimation fails
func (sde *StandardDensityEstimator) EstimateDensity(detections []postprocess.Result) (int, error) {
	count := 0
	for _, d := range detections {
		area := float64((d.Box.X2 - d.Box.X1) * (d.Box.Y2 - d.Box.Y1))
		if area >= float64(sde.MinBoxArea) {
			count++
		}
	}
	return count, nil
}

// GetDensityMetrics provides basic metrics for the standard estimator.
//
// Arguments:
//   - detections: The detections to analyze
//
// Returns:
//   - *DensityMetrics: Basic density metrics
//   - error: An error if analysis fails
func (sde *StandardDensityEstimator) GetDensityMetrics(
	detections []postprocess.Result,
) (*DensityMetrics, error) {
	metrics := &DensityMetrics{
		TotalObjects: len(detections),
	}

	validObjects := 0
	var totalArea float64

	for _, detection := range detections {
		area := float64((detection.Box.X2 - detection.Box.X1) * (detection.Box.Y2 - detection.Box.Y1))
		if area >= float64(sde.MinBoxArea) {
			validObjects++
			totalArea += area
		}
	}

	if validObjects > 0 {
		metrics.AverageObjectSize = totalArea / float64(validObjects)
	}

	return metrics, nil
}

// Package postprocess - provides Non-Maximum Suppression for detection results.
package postprocess

import (
	"sync"

	"github.com/nvr-ai/go-ml/images"
)

// NMSConfig defines parameters for Non-Maximum Suppression.
type NMSConfig struct {
	Greedy       bool    // If true, use greedy NMS.
	IoUThreshold float32 // Overlap threshold for suppression.
	ClassAware   bool    // If true, suppress only within same class.
	NumWorkers   int     // Number of goroutines for parallel IoU computation.
}

// ApplyNMS filters overlapping detections using Non-Maximum Suppression.
//
// Arguments:
//   - detections: Sorted slice of detections (highest score first).
//
// - config: NMS configuration. If true, suppress only within same class. If false, suppress all
// overlapping detections.
//
// Returns:
//   - Filtered slice of detections. If no detections are provided, returns nil.
func ApplyNMS(detections []Result, config *NMSConfig) []Result {
	n := len(detections)
	if n == 0 {
		return nil
	}

	used := make([]bool, n)
	filtered := make([]Result, 0, n)

	// Worker pool for parallel IoU computation.
	type job struct {
		i, j int
	}
	jobs := make(chan job, n*n)
	results := make(chan int, n*n)

	var wg sync.WaitGroup
	for w := 0; w < config.NumWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range jobs {
				if used[task.j] {
					continue
				}
				if config.ClassAware && detections[task.i].Class != detections[task.j].Class {
					continue
				}
				if images.CalculateIoU(
					detections[task.i].Box.X1,
					detections[task.i].Box.Y1,
					detections[task.i].Box.X2,
					detections[task.i].Box.Y2,
					detections[task.j].Box.X1,
					detections[task.j].Box.Y1,
					detections[task.j].Box.X2,
					detections[task.j].Box.Y2,
				) > config.IoUThreshold {
					results <- task.j
				}
			}
		}()
	}

	for i := 0; i < n; i++ {
		if used[i] {
			continue
		}
		filtered = append(filtered, detections[i])
		used[i] = true

		// Dispatch jobs to suppress overlapping boxes.
		for j := i + 1; j < n; j++ {
			if !used[j] {
				jobs <- job{i, j}
			}
		}

		// Drain suppression results.
		go func() {
			for j := range results {
				used[j] = true
			}
		}()
	}

	close(jobs)
	wg.Wait()
	close(results)

	return filtered
}

// ApplyGreedyNMS performs standard greedy Non-Maximum Suppression.
//
// Arguments:
//   - detections: Slice of detections sorted by descending confidence.
//   - iouThreshold: IoU threshold above which overlapping boxes are suppressed.
//
// Returns:
//   - Filtered slice of detections.
func ApplyGreedyNMS(detections []Result, config *NMSConfig) []Result {
	n := len(detections)
	if n == 0 {
		return nil
	}

	filtered := make([]Result, 0, n)
	used := make([]bool, n)

	for i := 0; i < n; i++ {
		if used[i] {
			continue
		}

		anchor := detections[i]
		filtered = append(filtered, anchor)
		used[i] = true

		for j := i + 1; j < n; j++ {
			if used[j] {
				continue
			}

			// Suppress if IoU exceeds threshold
			if images.CalculateIoU(
				anchor.Box.X1,
				anchor.Box.Y1,
				anchor.Box.X2,
				anchor.Box.Y2,
				detections[j].Box.X1,
				detections[j].Box.Y1,
				detections[j].Box.X2,
				detections[j].Box.Y2,
			) > config.IoUThreshold {
				used[j] = true
			}
		}
	}

	return filtered
}

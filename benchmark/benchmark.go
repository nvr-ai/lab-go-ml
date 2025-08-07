// Package benchmark - Functionality for running benchmarks.
package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/nvr-ai/go-ml/inference"
)

// Config represents the overall benchmark configuration
type Config struct {
	OutputDir       string               `json:"output_dir"`
	TestImagesPath  string               `json:"test_images_path"`
	Engine          inference.EngineType `json:"engine"`
	ModelPaths      map[string]string    `json:"model_paths"`
	MaxConcurrency  int                  `json:"max_concurrency"`
	TimeoutSeconds  int                  `json:"timeout_seconds"`
	SaveDetailedLog bool                 `json:"save_detailed_log"`
}

// DefaultConfig returns a default benchmark configuration
func DefaultConfig() *Config {
	return &Config{
		OutputDir:       "./benchmark_results",
		TestImagesPath:  "./test_images",
		ModelPaths:      make(map[string]string),
		MaxConcurrency:  1,
		TimeoutSeconds:  3600, // 1 hour
		SaveDetailedLog: true,
	}
}

// SaveConfig saves the benchmark configuration to a JSON file
func (bc *Config) SaveConfig(filename string) error {
	data, err := json.MarshalIndent(bc, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(filename, data, 0o644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// LoadConfig loads benchmark configuration from a JSON file
func LoadConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

func (bs *Suite) countDetections(result interface{}) int {
	// Use the ONNX engine's detection counting function
	return CountDetections(result)
}

// CountDetections counts the number of detections in an inference result
func CountDetections(result interface{}) int {
	if result == nil {
		return 0
	}

	// Handle different result types that might be returned by inference engines
	switch detections := result.(type) {
	case []interface{}:
		return len(detections)
	case []map[string]interface{}:
		return len(detections)
	default:
		// For unknown types, try to get a count if possible
		return 1 // Placeholder - actual implementation would depend on the specific result format
	}
}

// SaveResults persists benchmark results to filesystem
func (bs *Suite) SaveResults() error {
	bs.mu.RLock()
	results := make([]PerformanceMetrics, len(bs.results))
	copy(results, bs.results)
	bs.mu.RUnlock()

	// Ensure output directory exists
	if err := os.MkdirAll(bs.outputDir, 0o755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Save detailed results as JSON
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	resultsFile := filepath.Join(bs.outputDir, fmt.Sprintf("benchmark_results_%s.json", timestamp))

	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal results: %w", err)
	}

	if err := os.WriteFile(resultsFile, data, 0o644); err != nil {
		return fmt.Errorf("failed to write results file: %w", err)
	}

	// Save summary CSV
	summaryFile := filepath.Join(bs.outputDir, fmt.Sprintf("benchmark_summary_%s.csv", timestamp))
	if err := bs.saveSummaryCSV(summaryFile, results); err != nil {
		return fmt.Errorf("failed to save summary CSV: %w", err)
	}

	fmt.Printf("Results saved to: %s\n", resultsFile)
	fmt.Printf("Summary saved to: %s\n", summaryFile)

	return nil
}

func (bs *Suite) saveSummaryCSV(filename string, results []PerformanceMetrics) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write CSV header
	header := "Scenario,Model,Resolution,Format,FPS,Total_Duration_ms,Avg_Memory_MB,Detections,Error_Rate\n"
	if _, err := file.WriteString(header); err != nil {
		return err
	}

	// Write data rows
	for _, result := range results {
		avgMemoryMB := float64(result.MemoryStats.AllocBytes) / (1024 * 1024)
		line := fmt.Sprintf("%s,%s,%s,%s,%.2f,%.2f,%.2f,%d,%.4f\n",
			result.Scenario.Name,
			result.Scenario.ModelType,
			result.Scenario.Resolution.Name,
			result.Scenario.ImageFormat,
			result.FramesPerSecond,
			float64(result.TotalDuration.Nanoseconds())/1e6,
			avgMemoryMB,
			result.DetectionCount,
			result.ErrorRate,
		)
		if _, err := file.WriteString(line); err != nil {
			return err
		}
	}

	return nil
}

// GetResults returns all benchmark results
func (bs *Suite) GetResults() []PerformanceMetrics {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	results := make([]PerformanceMetrics, len(bs.results))
	copy(results, bs.results)
	return results
}

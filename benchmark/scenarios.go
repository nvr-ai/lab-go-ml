// Package benchmark - Functionality for running benchmarks.
package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/inference"
	"github.com/nvr-ai/go-ml/models/model"
)

// Scenario defines a specific test configuration
type Scenario struct {
	Name        string               `json:"name"`
	Model       model.Config         `json:"model"`
	EngineType  inference.EngineType `json:"engine_type"`
	Resolution  images.Resolution    `json:"resolution"`
	ImageFormat images.ImageFormat   `json:"image_format"`
	BatchSize   int                  `json:"batch_size"`
	Iterations  int                  `json:"iterations"`
	WarmupRuns  int                  `json:"warmup_runs"`
}

// ScenarioBuilder helps build test scenarios with fluent API
type ScenarioBuilder struct {
	scenario Scenario
}

// NewScenarioBuilder creates a new scenario builder
func NewScenarioBuilder(name string) *ScenarioBuilder {
	return &ScenarioBuilder{
		scenario: Scenario{
			Name:       name,
			BatchSize:  1,
			Iterations: 100,
			WarmupRuns: 10,
		},
	}
}

// WithEngineType sets the engine type
func (sb *ScenarioBuilder) WithEngineType(engineType inference.EngineType) *ScenarioBuilder {
	sb.scenario.EngineType = engineType
	return sb
}

// WithModel sets the model configuration
func (sb *ScenarioBuilder) WithModel(model model.Config) *ScenarioBuilder {
	sb.scenario.Model = model
	return sb
}

// WithResolution sets the image resolution
func (sb *ScenarioBuilder) WithResolution(width, height int) *ScenarioBuilder {
	sb.scenario.Resolution = images.Resolution{
		Width:  width,
		Height: height,
	}
	return sb
}

// WithImageFormat sets the image format
func (sb *ScenarioBuilder) WithImageFormat(format images.ImageFormat) *ScenarioBuilder {
	sb.scenario.ImageFormat = format
	return sb
}

// WithIterations sets the number of test iterations
func (sb *ScenarioBuilder) WithIterations(iterations int) *ScenarioBuilder {
	sb.scenario.Iterations = iterations
	return sb
}

// WithWarmupRuns sets the number of warmup runs
func (sb *ScenarioBuilder) WithWarmupRuns(warmups int) *ScenarioBuilder {
	sb.scenario.WarmupRuns = warmups
	return sb
}

// WithBatchSize sets the batch size for processing
func (sb *ScenarioBuilder) WithBatchSize(batchSize int) *ScenarioBuilder {
	sb.scenario.BatchSize = batchSize
	return sb
}

// Build returns the configured test scenario
func (sb *ScenarioBuilder) Build() Scenario {
	return sb.scenario
}

// ScenarioSet represents a collection of related test scenarios
type ScenarioSet struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Scenarios   []Scenario `json:"scenarios"`
}

// PredefinedScenarios contains common benchmark scenario sets
type PredefinedScenarios struct{}

// GetComprehensiveScenarios returns a comprehensive set of benchmark scenarios
func (ps *PredefinedScenarios) GetComprehensiveScenarios(models []model.Config) *ScenarioSet {
	scenarios := make([]Scenario, 0)

	// Test different resolutions for each model and format combination
	for _, m := range models {
		for _, resolution := range images.Resolutions {
			for _, format := range []images.ImageFormat{images.FormatJPEG, images.FormatWebP, images.FormatPNG} {
				scenario := NewScenarioBuilder(fmt.Sprintf("%s_%s_%s", m.Family, resolution.Alias, format)).
					WithModel(m).
					WithResolution(resolution.Pixels.Width, resolution.Pixels.Height).
					WithImageFormat(format).
					WithIterations(100).
					WithWarmupRuns(10).
					Build()

				scenarios = append(scenarios, scenario)
			}
		}
	}

	return &ScenarioSet{
		Name:        "Comprehensive Performance Test",
		Description: "Tests all combinations of models, resolutions, and image formats",
		Scenarios:   scenarios,
	}
}

// GetQuickScenarios returns a smaller set for quick testing
func (ps *PredefinedScenarios) GetQuickScenarios(models []model.Config) *ScenarioSet {
	scenarios := make([]Scenario, 0)

	// Quick test with common configurations
	commonResolutions := []images.Resolution{
		images.Resolutions[images.ResolutionAlias1MP],
		images.Resolutions[images.ResolutionAlias4MP],
	}

	for _, model := range models {
		for _, resolution := range commonResolutions {
			// Test only JPEG for quick scenarios
			scenario := NewScenarioBuilder(fmt.Sprintf("quick_%s_%s", model.Family, resolution.Alias)).
				WithModel(model).
				WithResolution(resolution.Width, resolution.Height).
				WithImageFormat(images.FormatJPEG).
				WithIterations(50).
				WithWarmupRuns(5).
				Build()

			scenarios = append(scenarios, scenario)
		}
	}

	return &ScenarioSet{
		Name:        "Quick Performance Test",
		Description: "Quick test with common configurations",
		Scenarios:   scenarios,
	}
}

// GetResolutionComparisonScenarios tests different resolutions with the same model
func (ps *PredefinedScenarios) GetResolutionComparisonScenarios(model model.Config) *ScenarioSet {
	scenarios := make([]Scenario, 0)

	for _, resolution := range images.Resolutions {
		scenario := NewScenarioBuilder(fmt.Sprintf("resolution_%s_%s", model.Family, resolution.Alias)).
			WithModel(model).
			WithResolution(resolution.Pixels.Width, resolution.Pixels.Height).
			WithImageFormat(images.FormatJPEG).
			WithIterations(100).
			WithWarmupRuns(10).
			Build()

		scenarios = append(scenarios, scenario)
	}

	return &ScenarioSet{
		Name:        fmt.Sprintf("Resolution Comparison - %s", model.Family),
		Description: fmt.Sprintf("Compares different input resolutions for %s model", model.Family),
		Scenarios:   scenarios,
	}
}

// GetFormatComparisonScenarios tests different image formats with the same model and resolution
func (ps *PredefinedScenarios) GetFormatComparisonScenarios(
	model model.Config,
	resolution images.Resolution,
) *ScenarioSet {
	scenarios := make([]Scenario, 0)

	formats := []images.ImageFormat{images.FormatJPEG, images.FormatWebP, images.FormatPNG}
	for _, format := range formats {
		scenario := NewScenarioBuilder(
			fmt.Sprintf("format_%s_%s_%s", model.Family, resolution.Alias, format),
		).
			WithModel(model).
			WithResolution(resolution.Width, resolution.Height).
			WithImageFormat(format).
			WithIterations(100).
			WithWarmupRuns(10).
			Build()

		scenarios = append(scenarios, scenario)
	}

	return &ScenarioSet{
		Name: fmt.Sprintf("Format Comparison - %s @ %s", model.Family, resolution.Alias),
		Description: fmt.Sprintf(
			"Compares different image formats for %s model at %s resolution",
			model.Family,
			resolution.Alias,
		),
		Scenarios: scenarios,
	}
}

// GetModelComparisonScenarios compares different models with the same configuration
func (ps *PredefinedScenarios) GetModelComparisonScenarios(
	models []model.Config,
	resolution images.Resolution,
	format images.ImageFormat,
) *ScenarioSet {
	scenarios := make([]Scenario, 0)

	for _, model := range models {
		scenario := NewScenarioBuilder(
			fmt.Sprintf("model_%s_%s_%s", model.Family, resolution.Alias, format),
		).
			WithModel(model).
			WithResolution(resolution.Width, resolution.Height).
			WithImageFormat(format).
			WithIterations(100).
			WithWarmupRuns(10).
			Build()

		scenarios = append(scenarios, scenario)
	}

	return &ScenarioSet{
		Name: fmt.Sprintf("Model Comparison @ %s %s", resolution.Alias, format),
		Description: fmt.Sprintf(
			"Compares different models at %s resolution with %s format",
			resolution.Alias,
			format,
		),
		Scenarios: scenarios,
	}
}

// SaveScenarioSet saves a scenario set to a JSON file
func SaveScenarioSet(scenarioSet *ScenarioSet, filename string) error {
	data, err := json.MarshalIndent(scenarioSet, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal scenario set: %w", err)
	}

	if err := os.WriteFile(filename, data, 0o644); err != nil {
		return fmt.Errorf("failed to write scenario file: %w", err)
	}

	return nil
}

// LoadScenarioSet loads a scenario set from a JSON file
func LoadScenarioSet(filename string) (*ScenarioSet, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read scenario file: %w", err)
	}

	var scenarioSet ScenarioSet
	if err := json.Unmarshal(data, &scenarioSet); err != nil {
		return nil, fmt.Errorf("failed to unmarshal scenario set: %w", err)
	}

	return &scenarioSet, nil
}

// RunAllScenarios executes all configured benchmark scenarios
func (bs *Suite) RunAllScenarios(ctx context.Context) error {
	bs.mu.Lock()
	scenarios := make([]Scenario, len(bs.scenarios))
	copy(scenarios, bs.scenarios)
	bs.mu.Unlock()

	for _, scenario := range scenarios {
		metrics, err := bs.RunScenario(ctx, scenario)
		if err != nil {
			fmt.Printf("Scenario %s failed: %v\n", scenario.Name, err)
			continue
		}

		bs.mu.Lock()
		bs.results = append(bs.results, *metrics)
		bs.mu.Unlock()

		fmt.Printf("Scenario %s completed: %.2f FPS\n", scenario.Name, metrics.FramesPerSecond)
	}

	return bs.SaveResults()
}

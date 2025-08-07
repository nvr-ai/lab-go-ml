package benchmark

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/inference"
)

// ScenarioBuilder helps build test scenarios with fluent API
type ScenarioBuilder struct {
	scenario TestScenario
}

// NewScenarioBuilder creates a new scenario builder
func NewScenarioBuilder(name string) *ScenarioBuilder {
	return &ScenarioBuilder{
		scenario: TestScenario{
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
func (sb *ScenarioBuilder) WithModel(modelType ModelType, modelPath string) *ScenarioBuilder {
	sb.scenario.ModelType = modelType
	sb.scenario.ModelPath = modelPath
	return sb
}

// WithResolution sets the image resolution
func (sb *ScenarioBuilder) WithResolution(width, height int) *ScenarioBuilder {
	sb.scenario.Resolution = Resolution{
		Width:  width,
		Height: height,
		Name:   fmt.Sprintf("%dx%d", width, height),
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
func (sb *ScenarioBuilder) Build() TestScenario {
	return sb.scenario
}

// ScenarioSet represents a collection of related test scenarios
type ScenarioSet struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Scenarios   []TestScenario `json:"scenarios"`
}

// PredefinedScenarios contains common benchmark scenario sets
type PredefinedScenarios struct{}

// GetComprehensiveScenarios returns a comprehensive set of benchmark scenarios
func (ps *PredefinedScenarios) GetComprehensiveScenarios(modelPaths map[ModelType]string) *ScenarioSet {
	scenarios := make([]TestScenario, 0)

	// Test different resolutions for each model and format combination
	for modelType, modelPath := range modelPaths {
		for _, resolution := range images.GetAllResolutions() {
			for _, format := range []images.ImageFormat{images.FormatJPEG, images.FormatWebP, images.FormatPNG} {
				scenario := NewScenarioBuilder(fmt.Sprintf("%s_%s_%s", modelType, resolution.Type, format)).
					WithModel(modelType, modelPath).
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
func (ps *PredefinedScenarios) GetQuickScenarios(modelPaths map[ModelType]string) *ScenarioSet {
	scenarios := make([]TestScenario, 0)

	// Quick test with common configurations
	commonResolutions := []Resolution{
		{Width: 416, Height: 416, Name: "416x416"},
		{Width: 640, Height: 640, Name: "640x640"},
	}

	for modelType, modelPath := range modelPaths {
		for _, resolution := range commonResolutions {
			// Test only JPEG for quick scenarios
			scenario := NewScenarioBuilder(fmt.Sprintf("quick_%s_%s", modelType, resolution.Name)).
				WithModel(modelType, modelPath).
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
func (ps *PredefinedScenarios) GetResolutionComparisonScenarios(modelType ModelType, modelPath string) *ScenarioSet {
	scenarios := make([]TestScenario, 0)

	for _, resolution := range images.GetAllResolutions() {
		scenario := NewScenarioBuilder(fmt.Sprintf("resolution_%s_%s", modelType, resolution.Type)).
			WithModel(modelType, modelPath).
			WithResolution(resolution.Pixels.Width, resolution.Pixels.Height).
			WithImageFormat(images.FormatJPEG). // Use consistent format
			WithIterations(100).
			WithWarmupRuns(10).
			Build()

		scenarios = append(scenarios, scenario)
	}

	return &ScenarioSet{
		Name:        fmt.Sprintf("Resolution Comparison - %s", modelType),
		Description: fmt.Sprintf("Compares different input resolutions for %s model", modelType),
		Scenarios:   scenarios,
	}
}

// GetFormatComparisonScenarios tests different image formats with the same model and resolution
func (ps *PredefinedScenarios) GetFormatComparisonScenarios(modelType ModelType, modelPath string, resolution Resolution) *ScenarioSet {
	scenarios := make([]TestScenario, 0)

	formats := []images.ImageFormat{images.FormatJPEG, images.FormatWebP, images.FormatPNG}
	for _, format := range formats {
		scenario := NewScenarioBuilder(fmt.Sprintf("format_%s_%s_%s", modelType, resolution.Name, format)).
			WithModel(modelType, modelPath).
			WithResolution(resolution.Width, resolution.Height).
			WithImageFormat(format).
			WithIterations(100).
			WithWarmupRuns(10).
			Build()

		scenarios = append(scenarios, scenario)
	}

	return &ScenarioSet{
		Name:        fmt.Sprintf("Format Comparison - %s @ %s", modelType, resolution.Name),
		Description: fmt.Sprintf("Compares different image formats for %s model at %s resolution", modelType, resolution.Name),
		Scenarios:   scenarios,
	}
}

// GetModelComparisonScenarios compares different models with the same configuration
func (ps *PredefinedScenarios) GetModelComparisonScenarios(modelPaths map[ModelType]string, resolution Resolution, format images.ImageFormat) *ScenarioSet {
	scenarios := make([]TestScenario, 0)

	for modelType, modelPath := range modelPaths {
		scenario := NewScenarioBuilder(fmt.Sprintf("model_%s_%s_%s", modelType, resolution.Name, format)).
			WithModel(modelType, modelPath).
			WithResolution(resolution.Width, resolution.Height).
			WithImageFormat(format).
			WithIterations(100).
			WithWarmupRuns(10).
			Build()

		scenarios = append(scenarios, scenario)
	}

	return &ScenarioSet{
		Name:        fmt.Sprintf("Model Comparison @ %s %s", resolution.Name, format),
		Description: fmt.Sprintf("Compares different models at %s resolution with %s format", resolution.Name, format),
		Scenarios:   scenarios,
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

// BenchmarkConfig represents the overall benchmark configuration
type BenchmarkConfig struct {
	OutputDir       string               `json:"output_dir"`
	TestImagesPath  string               `json:"test_images_path"`
	Engine          inference.EngineType `json:"engine"`
	ModelPaths      map[string]string    `json:"model_paths"`
	MaxConcurrency  int                  `json:"max_concurrency"`
	TimeoutSeconds  int                  `json:"timeout_seconds"`
	SaveDetailedLog bool                 `json:"save_detailed_log"`
}

// DefaultBenchmarkConfig returns a default benchmark configuration
func DefaultBenchmarkConfig() *BenchmarkConfig {
	return &BenchmarkConfig{
		OutputDir:       "./benchmark_results",
		TestImagesPath:  "./test_images",
		ModelPaths:      make(map[string]string),
		MaxConcurrency:  1,
		TimeoutSeconds:  3600, // 1 hour
		SaveDetailedLog: true,
	}
}

// SaveConfig saves the benchmark configuration to a JSON file
func (bc *BenchmarkConfig) SaveConfig(filename string) error {
	data, err := json.MarshalIndent(bc, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(filename, data, 0o644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// LoadBenchmarkConfig loads benchmark configuration from a JSON file
func LoadBenchmarkConfig(filename string) (*BenchmarkConfig, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config BenchmarkConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

package benchmark

import (
	"context"
	"image"
	"testing"

	"github.com/stretchr/testify/assert"
)

// MockInferenceEngine for testing
type MockInferenceEngine struct {
	loadModelCalled bool
	predictCalled   bool
	closeCalled     bool
	predictResult   interface{}
	predictError    error
}

func (m *MockInferenceEngine) LoadModel(modelPath string, config map[string]interface{}) error {
	m.loadModelCalled = true
	return nil
}

func (m *MockInferenceEngine) Predict(ctx context.Context, img image.Image) (interface{}, error) {
	m.predictCalled = true
	return m.predictResult, m.predictError
}

func (m *MockInferenceEngine) Close() error {
	m.closeCalled = true
	return nil
}

func (m *MockInferenceEngine) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model_type": "mock",
		"version":    "1.0",
	}
}

func TestNewBenchmarkSuite(t *testing.T) {
	engine := &MockInferenceEngine{}
	outputDir := "./test_output"
	
	suite := NewBenchmarkSuite(engine, outputDir)
	
	assert.NotNil(t, suite)
	assert.Equal(t, engine, suite.engine)
	assert.Equal(t, outputDir, suite.outputDir)
	assert.Empty(t, suite.scenarios)
	assert.Empty(t, suite.results)
}

func TestScenarioBuilder(t *testing.T) {
	builder := NewScenarioBuilder("test_scenario")
	
	scenario := builder.
		WithModel(ModelYOLO, "./test_model.onnx").
		WithResolution(416, 416).
		WithImageFormat(FormatJPEG).
		WithIterations(50).
		WithWarmupRuns(5).
		WithBatchSize(2).
		Build()
	
	assert.Equal(t, "test_scenario", scenario.Name)
	assert.Equal(t, ModelYOLO, scenario.ModelType)
	assert.Equal(t, "./test_model.onnx", scenario.ModelPath)
	assert.Equal(t, 416, scenario.Resolution.Width)
	assert.Equal(t, 416, scenario.Resolution.Height)
	assert.Equal(t, FormatJPEG, scenario.ImageFormat)
	assert.Equal(t, 50, scenario.Iterations)
	assert.Equal(t, 5, scenario.WarmupRuns)
	assert.Equal(t, 2, scenario.BatchSize)
}

func TestAddScenario(t *testing.T) {
	engine := &MockInferenceEngine{}
	suite := NewBenchmarkSuite(engine, "./test_output")
	
	scenario := NewScenarioBuilder("test").
		WithModel(ModelYOLO, "./model.onnx").
		WithResolution(224, 224).
		WithImageFormat(FormatJPEG).
		Build()
	
	suite.AddScenario(scenario)
	
	assert.Len(t, suite.scenarios, 1)
	assert.Equal(t, scenario, suite.scenarios[0])
}

func TestPredefinedScenarios(t *testing.T) {
	predefined := &PredefinedScenarios{}
	modelPaths := map[ModelType]string{
		ModelYOLO: "./yolo.onnx",
	}
	
	// Test quick scenarios
	quick := predefined.GetQuickScenarios(modelPaths)
	assert.NotEmpty(t, quick.Scenarios)
	assert.Equal(t, "Quick Performance Test", quick.Name)
	
	// Test comprehensive scenarios
	comprehensive := predefined.GetComprehensiveScenarios(modelPaths)
	assert.NotEmpty(t, comprehensive.Scenarios)
	assert.Equal(t, "Comprehensive Performance Test", comprehensive.Name)
	
	// Test resolution comparison
	resolution := predefined.GetResolutionComparisonScenarios(ModelYOLO, "./yolo.onnx")
	assert.NotEmpty(t, resolution.Scenarios)
	assert.Contains(t, resolution.Name, "Resolution Comparison")
	
	// Test format comparison
	testRes := Resolution{Width: 416, Height: 416, Name: "416x416"}
	format := predefined.GetFormatComparisonScenarios(ModelYOLO, "./yolo.onnx", testRes)
	assert.NotEmpty(t, format.Scenarios)
	assert.Contains(t, format.Name, "Format Comparison")
}

func TestCountDetections(t *testing.T) {
	// Test nil result
	count := CountDetections(nil)
	assert.Equal(t, 0, count)
	
	// Test slice of interfaces
	detections := []interface{}{
		map[string]interface{}{"bbox": []float64{1, 2, 3, 4}},
		map[string]interface{}{"bbox": []float64{5, 6, 7, 8}},
	}
	count = CountDetections(detections)
	assert.Equal(t, 2, count)
	
	// Test slice of maps
	mapDetections := []map[string]interface{}{
		{"bbox": []float64{1, 2, 3, 4}},
		{"bbox": []float64{5, 6, 7, 8}},
		{"bbox": []float64{9, 10, 11, 12}},
	}
	count = CountDetections(mapDetections)
	assert.Equal(t, 3, count)
}

func TestBenchmarkConfig(t *testing.T) {
	// Test default config
	config := DefaultBenchmarkConfig()
	assert.NotNil(t, config)
	assert.Equal(t, "./benchmark_results", config.OutputDir)
	assert.Equal(t, "./test_images", config.TestImagesPath)
	assert.NotNil(t, config.ModelPaths)
	assert.Equal(t, 1, config.MaxConcurrency)
	assert.Equal(t, 3600, config.TimeoutSeconds)
	assert.True(t, config.SaveDetailedLog)
}

// Test basic functionality without ONNX dependencies
func TestEngineInterface(t *testing.T) {
	// Test that our mock engine works
	engine := &MockInferenceEngine{}
	suite := NewBenchmarkSuite(engine, "./test_output")
	assert.NotNil(t, suite)
}

// Benchmark test for the framework itself
func BenchmarkScenarioCreation(b *testing.B) {
	predefined := &PredefinedScenarios{}
	modelPaths := map[ModelType]string{
		ModelYOLO: "./test.onnx",
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = predefined.GetQuickScenarios(modelPaths)
	}
}

func BenchmarkScenarioBuilder(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewScenarioBuilder("test").
			WithModel(ModelYOLO, "./model.onnx").
			WithResolution(416, 416).
			WithImageFormat(FormatJPEG).
			WithIterations(100).
			Build()
	}
}
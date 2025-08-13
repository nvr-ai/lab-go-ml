package main

import (
	"fmt"
	"log"

	"github.com/nvr-ai/go-ml/benchmark"
	"github.com/nvr-ai/go-ml/images"
	"github.com/nvr-ai/go-ml/inference"
)

// Example program to create and save benchmark scenarios
func main() {
	predefined := &benchmark.PredefinedScenarios{}

	// Define model paths
	modelPaths := map[benchmark.ModelType]string{
		benchmark.ModelYOLO: "../data/yolov8n.onnx",
	}

	// Create comprehensive scenarios
	comprehensive := predefined.GetComprehensiveScenarios(modelPaths)
	err := benchmark.SaveScenarioSet(comprehensive, "comprehensive_scenarios.json")
	if err != nil {
		log.Fatalf("Failed to save comprehensive scenarios: %v", err)
	}
	fmt.Printf("Saved %d comprehensive scenarios\n", len(comprehensive.Scenarios))

	// Create quick scenarios
	quick := predefined.GetQuickScenarios(modelPaths)
	err = benchmark.SaveScenarioSet(quick, "quick_scenarios.json")
	if err != nil {
		log.Fatalf("Failed to save quick scenarios: %v", err)
	}
	fmt.Printf("Saved %d quick scenarios\n", len(quick.Scenarios))

	// Create resolution comparison scenarios
	resolutions := predefined.GetResolutionComparisonScenarios(
		benchmark.ModelYOLO,
		"../data/yolov8n.onnx",
	)
	err = benchmark.SaveScenarioSet(resolutions, "resolution_scenarios.json")
	if err != nil {
		log.Fatalf("Failed to save resolution scenarios: %v", err)
	}
	fmt.Printf("Saved %d resolution scenarios\n", len(resolutions.Scenarios))

	// Create format comparison scenarios
	resolution416 := benchmark.Resolution{Width: 416, Height: 416, Name: "416x416"}
	formats := predefined.GetFormatComparisonScenarios(
		benchmark.ModelYOLO,
		"../data/yolov8n.onnx",
		resolution416,
	)
	err = benchmark.SaveScenarioSet(formats, "format_scenarios.json")
	if err != nil {
		log.Fatalf("Failed to save format scenarios: %v", err)
	}
	fmt.Printf("Saved %d format scenarios\n", len(formats.Scenarios))

	// Create custom scenario using builder
	customScenario := benchmark.NewScenarioBuilder("custom_high_res_webp").
		WithModel(benchmark.ModelYOLO, "../data/yolov8n.onnx").
		WithEngineType(inference.EngineOpenVINO).
		WithResolution(1024, 1024).
		WithImageFormat(images.FormatWebP).
		WithIterations(50).
		WithWarmupRuns(5).
		Build()

	customSet := &benchmark.ScenarioSet{
		Name:        "Custom High Resolution WebP Test",
		Description: "Tests high resolution WebP images with YOLO model",
		Scenarios:   []benchmark.Scenario{customScenario},
	}

	err = benchmark.SaveScenarioSet(customSet, "custom_scenarios.json")
	if err != nil {
		log.Fatalf("Failed to save custom scenarios: %v", err)
	}
	fmt.Printf("Saved %d custom scenarios\n", len(customSet.Scenarios))

	fmt.Println("All scenario files created successfully!")
}

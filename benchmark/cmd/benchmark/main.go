package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/nvr-ai/go-ml/benchmark"
	"github.com/nvr-ai/go-ml/inference"
	"github.com/nvr-ai/go-ml/models/model"
)

func main() {
	var (
		configFile   = flag.String("config", "", "Path to benchmark configuration file")
		scenarioFile = flag.String("scenarios", "", "Path to scenario configuration file")
		outputDir    = flag.String("output", "./benchmark_results", "Output directory for results")
		testImages   = flag.String("images", "", "Path to test images directory or file")
		modelPath    = flag.String("model", "", "Path to ONNX model file")
		// engineType    = flag.String("engine", "onnx", "Engine type to use for inference")
		quick         = flag.Bool("quick", false, "Run quick benchmark scenarios")
		comprehensive = flag.Bool("comprehensive", false, "Run comprehensive benchmark scenarios")
		resolutions   = flag.Bool("resolutions", false, "Compare different input resolutions")
		formats       = flag.Bool("formats", false, "Compare different image formats")
		timeout       = flag.Duration("timeout", 30*time.Minute, "Benchmark timeout duration")
	)
	flag.Parse()

	// Validate required parameters
	if *testImages == "" {
		log.Fatal("Test images path is required (-images)")
	}

	if *modelPath == "" && *configFile == "" {
		log.Fatal("Either model path (-model) or config file (-config) is required")
	}

	// Create benchmark suite
	suite := benchmark.NewSuite(benchmark.NewSuiteArgs{
		OutputPath: *outputDir,
		Engine:     inference.EngineONNX,
	})

	// Load configuration if provided
	var config *benchmark.Config
	if *configFile != "" {
		var err error
		config, err = benchmark.LoadConfig(*configFile)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	} else {
		config = benchmark.DefaultConfig()
		config.OutputDir = *outputDir
		config.TestImagesPath = *testImages
		if *modelPath != "" {
			config.ModelPaths = map[string]string{
				"yolo": *modelPath,
			}
		}
	}

	// Generate scenarios based on flags
	predefined := &benchmark.PredefinedScenarios{}
	modelPaths := make(map[model.Family]string)

	// Convert string map to ModelType map
	for key, path := range config.ModelPaths {
		switch key {
		case "yolo":
			modelPaths[model.ModelFamilyYOLO] = path
		case "d-fine", "dfine":
			modelPaths[model.ModelFamilyCOCO] = path
		}
	}

	// Add scenarios based on flags
	if *scenarioFile != "" {
		scenarioSet, err := benchmark.LoadScenarioSet(*scenarioFile)
		if err != nil {
			log.Fatalf("Failed to load scenario file: %v", err)
		}
		for _, scenario := range scenarioSet.Scenarios {
			suite.AddScenario(scenario)
		}
		fmt.Printf("Loaded %d scenarios from %s\n", len(scenarioSet.Scenarios), *scenarioFile)
	} else {
		// Add scenarios based on command line flags
		if *quick {
			scenarios := predefined.GetQuickScenarios(modelPaths)
			for _, scenario := range scenarios.Scenarios {
				suite.AddScenario(scenario)
			}
			fmt.Printf("Added %d quick scenarios\n", len(scenarios.Scenarios))
		}

		if *comprehensive {
			scenarios := predefined.GetComprehensiveScenarios(modelPaths)
			for _, scenario := range scenarios.Scenarios {
				suite.AddScenario(scenario)
			}
			fmt.Printf("Added %d comprehensive scenarios\n", len(scenarios.Scenarios))
		}

		if *resolutions {
			for modelType, modelPath := range modelPaths {
				scenarios := predefined.GetResolutionComparisonScenarios(modelType, modelPath)
				for _, scenario := range scenarios.Scenarios {
					suite.AddScenario(scenario)
				}
				fmt.Printf("Added %d resolution comparison scenarios for %s\n", len(scenarios.Scenarios), modelType)
			}
		}

		if *formats {
			for modelType, modelPath := range modelPaths {
				resolution := benchmark.Resolution{Width: 416, Height: 416, Name: "416x416"}
				scenarios := predefined.GetFormatComparisonScenarios(modelType, modelPath, resolution)
				for _, scenario := range scenarios.Scenarios {
					suite.AddScenario(scenario)
				}
				fmt.Printf("Added %d format comparison scenarios for %s\n", len(scenarios.Scenarios), modelType)
			}
		}

		// If no specific scenarios requested, use quick by default
		if !*quick && !*comprehensive && !*resolutions && !*formats {
			scenarios := predefined.GetQuickScenarios(modelPaths)
			for _, scenario := range scenarios.Scenarios {
				suite.AddScenario(scenario)
			}
			fmt.Printf("Added %d default quick scenarios\n", len(scenarios.Scenarios))
		}
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	// Run benchmarks
	fmt.Println("Starting benchmark execution...")
	start := time.Now()

	// Ensure cleanup happens
	defer engines.CleanupSessionCache()

	err = suite.RunAllScenarios(ctx)
	if err != nil {
		log.Fatalf("Benchmark execution failed: %v", err)
	}

	duration := time.Since(start)
	fmt.Printf("Benchmark completed in %v\n", duration)

	// Print summary
	results := suite.GetResults()
	fmt.Printf("\n=== BENCHMARK RESULTS SUMMARY ===\n")
	fmt.Printf("Total scenarios: %d\n", len(results))
	fmt.Printf("Results saved to: %s\n", *outputDir)

	// Find best performing scenario
	var bestFPS float64
	var bestScenario string
	for _, result := range results {
		if result.FramesPerSecond > bestFPS {
			bestFPS = result.FramesPerSecond
			bestScenario = result.Scenario.Name
		}
		fmt.Printf("  %s: %.2f FPS (%.2f MB memory)\n",
			result.Scenario.Name,
			result.FramesPerSecond,
			float64(result.MemoryStats.AllocBytes)/(1024*1024))
	}

	fmt.Printf("\nBest performing scenario: %s (%.2f FPS)\n", bestScenario, bestFPS)
}

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", filepath.Base(os.Args[0]))
		fmt.Fprintf(os.Stderr, "Benchmark tool for ML inference performance testing.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(
			os.Stderr,
			"  %s -images ./test_images -model ./yolov8n.onnx -quick\n",
			filepath.Base(os.Args[0]),
		)
		fmt.Fprintf(
			os.Stderr,
			"  %s -config ./benchmark_config.json -scenarios ./scenarios.json\n",
			filepath.Base(os.Args[0]),
		)
		fmt.Fprintf(
			os.Stderr,
			"  %s -images ./test_images -model ./yolov8n.onnx -resolutions -formats\n",
			filepath.Base(os.Args[0]),
		)
	}
}

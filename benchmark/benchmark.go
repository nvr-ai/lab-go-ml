package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/nvr-ai/go-ml/images"
)

// ImageFormat represents supported image formats for benchmarking
type ImageFormat string

const (
	FormatJPEG ImageFormat = "jpeg"
	FormatWebP ImageFormat = "webp"
	FormatPNG  ImageFormat = "png"
)

// ModelType represents different ML model types
type ModelType string

const (
	ModelYOLO  ModelType = "yolo"
	ModelDFine ModelType = "d-fine"
)

// Resolution represents image dimensions for benchmarking
type Resolution struct {
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Name   string `json:"name"`
}

// Common resolutions for benchmarking
var CommonResolutions = []Resolution{
	{Width: 224, Height: 224, Name: "224x224"},
	{Width: 416, Height: 416, Name: "416x416"},
	{Width: 512, Height: 512, Name: "512x512"},
	{Width: 640, Height: 640, Name: "640x640"},
	{Width: 1024, Height: 1024, Name: "1024x1024"},
}

// TestScenario defines a specific test configuration
type TestScenario struct {
	Name        string      `json:"name"`
	ModelType   ModelType   `json:"model_type"`
	ModelPath   string      `json:"model_path"`
	Resolution  Resolution  `json:"resolution"`
	ImageFormat ImageFormat `json:"image_format"`
	BatchSize   int         `json:"batch_size"`
	Iterations  int         `json:"iterations"`
	WarmupRuns  int         `json:"warmup_runs"`
}

// PerformanceMetrics captures detailed performance data
type PerformanceMetrics struct {
	Scenario            TestScenario  `json:"scenario"`
	Timestamp           time.Time     `json:"timestamp"`
	TotalDuration       time.Duration `json:"total_duration"`
	ImageResizeDuration time.Duration `json:"image_resize_duration"`
	InferenceDuration   time.Duration `json:"inference_duration"`
	PostProcessDuration time.Duration `json:"post_process_duration"`
	FramesPerSecond     float64       `json:"frames_per_second"`
	MemoryStats         MemoryMetrics `json:"memory_stats"`
	CPUStats            CPUMetrics    `json:"cpu_stats"`
	DiskIOStats         DiskIOMetrics `json:"disk_io_stats"`
	DetectionCount      int           `json:"detection_count"`
	ErrorRate           float64       `json:"error_rate"`
}

// MemoryMetrics captures memory usage statistics
type MemoryMetrics struct {
	AllocBytes      uint64 `json:"alloc_bytes"`
	TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	SysBytes        uint64 `json:"sys_bytes"`
	NumGC           uint32 `json:"num_gc"`
	HeapAllocBytes  uint64 `json:"heap_alloc_bytes"`
	HeapSysBytes    uint64 `json:"heap_sys_bytes"`
}

// CPUMetrics captures CPU usage statistics
type CPUMetrics struct {
	UserTime   time.Duration `json:"user_time"`
	SystemTime time.Duration `json:"system_time"`
	NumCPU     int           `json:"num_cpu"`
}

// DiskIOMetrics captures disk I/O statistics
type DiskIOMetrics struct {
	ReadBytes  uint64 `json:"read_bytes"`
	WriteBytes uint64 `json:"write_bytes"`
	ReadOps    uint64 `json:"read_ops"`
	WriteOps   uint64 `json:"write_ops"`
}

// InferenceEngine defines the interface for ML inference engines
type InferenceEngine interface {
	LoadModel(modelPath string, config map[string]interface{}) error
	Predict(ctx context.Context, img image.Image) (interface{}, error)
	Close() error
	GetModelInfo() map[string]interface{}
}

// BenchmarkSuite manages and executes benchmark scenarios
type BenchmarkSuite struct {
	scenarios   []TestScenario
	engine      InferenceEngine
	outputDir   string
	testImages  [][]byte
	imageFormat ImageFormat
	mu          sync.RWMutex
	results     []PerformanceMetrics
}

// NewBenchmarkSuite creates a new benchmark suite
func NewBenchmarkSuite(engine InferenceEngine, outputDir string) *BenchmarkSuite {
	return &BenchmarkSuite{
		engine:    engine,
		outputDir: outputDir,
		scenarios: make([]TestScenario, 0),
		results:   make([]PerformanceMetrics, 0),
	}
}

// AddScenario adds a test scenario to the benchmark suite
func (bs *BenchmarkSuite) AddScenario(scenario TestScenario) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.scenarios = append(bs.scenarios, scenario)
}

// LoadTestImages loads test images from a directory or file
func (bs *BenchmarkSuite) LoadTestImages(imagePath string, format ImageFormat) error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	// Check if path is a file or directory
	info, err := os.Stat(imagePath)
	if err != nil {
		return fmt.Errorf("failed to stat image path: %w", err)
	}

	if info.IsDir() {
		return bs.loadImagesFromDirectory(imagePath, format)
	}
	return bs.loadImageFromFile(imagePath, format)
}

func (bs *BenchmarkSuite) loadImagesFromDirectory(dirPath string, format ImageFormat) error {
	files, err := os.ReadDir(dirPath)
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	bs.testImages = make([][]byte, 0)
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		filePath := filepath.Join(dirPath, file.Name())
		if err := bs.loadImageFromFile(filePath, format); err != nil {
			continue // Skip files that can't be loaded
		}
	}

	if len(bs.testImages) == 0 {
		return fmt.Errorf("no valid images found in directory: %s", dirPath)
	}

	return nil
}

func (bs *BenchmarkSuite) loadImageFromFile(filePath string, format ImageFormat) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read image file: %w", err)
	}

	bs.testImages = append(bs.testImages, data)
	bs.imageFormat = format
	return nil
}

// RunScenario executes a single benchmark scenario
func (bs *BenchmarkSuite) RunScenario(ctx context.Context, scenario TestScenario) (*PerformanceMetrics, error) {
	// Load model
	modelConfig := map[string]interface{}{
		"input_shape":          []int{scenario.Resolution.Width, scenario.Resolution.Height},
		"confidence_threshold": 0.5,
		"nms_threshold":        0.4,
	}

	if err := bs.engine.LoadModel(scenario.ModelPath, modelConfig); err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}
	defer bs.engine.Close()

	metrics := &PerformanceMetrics{
		Scenario:  scenario,
		Timestamp: time.Now(),
	}

	// Warmup runs
	for i := 0; i < scenario.WarmupRuns; i++ {
		if len(bs.testImages) > 0 {
			testImg := bs.testImages[i%len(bs.testImages)]
			if _, err := bs.processImage(ctx, testImg, scenario); err != nil {
				continue // Skip warmup errors
			}
		}
	}

	// Capture initial memory stats
	var startMem runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&startMem)

	startTime := time.Now()
	totalDetections := 0
	errors := 0

	// Run benchmark iterations
	for i := 0; i < scenario.Iterations; i++ {
		if len(bs.testImages) == 0 {
			errors++
			continue
		}

		testImg := bs.testImages[i%len(bs.testImages)]

		detectionCount, err := bs.processImage(ctx, testImg, scenario)
		if err != nil {
			errors++
			continue
		}

		totalDetections += detectionCount
	}

	totalDuration := time.Since(startTime)

	// Capture final memory stats
	var endMem runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&endMem)

	// Calculate metrics
	metrics.TotalDuration = totalDuration
	metrics.FramesPerSecond = float64(scenario.Iterations) / totalDuration.Seconds()
	metrics.DetectionCount = totalDetections
	metrics.ErrorRate = float64(errors) / float64(scenario.Iterations)

	metrics.MemoryStats = MemoryMetrics{
		AllocBytes:      endMem.Alloc,
		TotalAllocBytes: endMem.TotalAlloc - startMem.TotalAlloc,
		SysBytes:        endMem.Sys,
		NumGC:           endMem.NumGC - startMem.NumGC,
		HeapAllocBytes:  endMem.HeapAlloc,
		HeapSysBytes:    endMem.HeapSys,
	}

	metrics.CPUStats = CPUMetrics{
		NumCPU: runtime.NumCPU(),
	}

	return metrics, nil
}

func (bs *BenchmarkSuite) processImage(ctx context.Context, imageData []byte, scenario TestScenario) (int, error) {
	// Resize image
	resizeStart := time.Now()

	var resizedImg image.Image
	var err error

	switch scenario.ImageFormat {
	case FormatJPEG:
		resizedImg, err = images.ResizeImageToImage(imageData, scenario.Resolution.Width, scenario.Resolution.Height, images.FormatJPEG)
	case FormatWebP:
		resizedImg, err = images.ResizeImageToImage(imageData, scenario.Resolution.Width, scenario.Resolution.Height, images.FormatWebP)
	case FormatPNG:
		resizedImg, err = images.ResizeImageToImage(imageData, scenario.Resolution.Width, scenario.Resolution.Height, images.FormatPNG)
	default:
		return 0, fmt.Errorf("unsupported image format: %s", scenario.ImageFormat)
	}

	if err != nil {
		return 0, fmt.Errorf("failed to resize image: %w", err)
	}

	resizeDuration := time.Since(resizeStart)

	// Run inference
	inferenceStart := time.Now()
	result, err := bs.engine.Predict(ctx, resizedImg)
	if err != nil {
		return 0, fmt.Errorf("inference failed: %w", err)
	}
	inferenceDuration := time.Since(inferenceStart)

	// Count detections (implementation depends on result format)
	detectionCount := bs.countDetections(result)

	// Store timing information (could be accumulated)
	_ = resizeDuration
	_ = inferenceDuration

	return detectionCount, nil
}

func (bs *BenchmarkSuite) countDetections(result interface{}) int {
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

// RunAllScenarios executes all configured benchmark scenarios
func (bs *BenchmarkSuite) RunAllScenarios(ctx context.Context) error {
	bs.mu.Lock()
	scenarios := make([]TestScenario, len(bs.scenarios))
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

// SaveResults persists benchmark results to filesystem
func (bs *BenchmarkSuite) SaveResults() error {
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

func (bs *BenchmarkSuite) saveSummaryCSV(filename string, results []PerformanceMetrics) error {
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
func (bs *BenchmarkSuite) GetResults() []PerformanceMetrics {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	results := make([]PerformanceMetrics, len(bs.results))
	copy(results, bs.results)
	return results
}

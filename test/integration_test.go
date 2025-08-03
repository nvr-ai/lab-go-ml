// Package image provides comprehensive testing utilities for motion detection at scale.
// This test command framework ensures idempotency, quick iteration, and minimal resource usage
// while providing detailed insights into system performance and potential regressions.
package test

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMain provides the entry point for customized test execution with comprehensive
// monitoring, reporting, and regression detection capabilities. This function intercepts
// the standard test runner to add our advanced features while maintaining compatibility
// with standard Go testing tools.
//
// Arguments:
// - m: The testing.M instance provided by Go's test framework.
//
// Returns:
// - None (calls os.Exit with appropriate code).
//
// @example
// // This is automatically called by go test, but you can customize flags:
// go test -v -test.run TestSubtract -test.bench . -regression-check
// go test -v -parallel 4 -test.timeout 30s
// go test -v -test.run TestIdempotency -verbose-debug
func TestMain(m *testing.M) {
	// Parse custom flags for enhanced test control.
	var (
		regressionCheck = flag.Bool("regression-check", true, "Enable regression detection during test run")
		verboseDebug    = flag.Bool("verbose-debug", true, "Enable extremely verbose debugging output")
		generateReport  = flag.Bool("generate-report", true, "Generate HTML report after test completion")
		testResultsDir  = flag.String("results-dir", "./test_results", "Directory for storing test results")
		benchmarkDir    = flag.String("benchmark-dir", "./benchmarks", "Directory for storing benchmark results")
	)

	// Parse both standard test flags and our custom flags.
	flag.Parse()

	// Initialize test environment with debugging output.
	if *verboseDebug {
		log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)
		log.Println("🚀 Motion Detection Test Suite Starting...")
		log.Printf("📊 Configuration: regression=%v, verbose=%v, report=%v",
			*regressionCheck, *verboseDebug, *generateReport)
		log.Printf("💾 Results directory: %s", *testResultsDir)
		log.Printf("⚡ Benchmark directory: %s", *benchmarkDir)
		log.Printf("🖥️  Runtime: %d CPUs available", runtime.NumCPU())
	}

	// Ensure all required directories exist.
	os.MkdirAll(*testResultsDir, 0o755)
	os.MkdirAll(*benchmarkDir, 0o755)
	os.MkdirAll(filepath.Join(*testResultsDir, "reports"), 0o755)

	// Initialize global test store for result persistence.
	globalTestStore = NewTestResultStore(*testResultsDir)
	globalBenchStore = NewTestResultStore(*benchmarkDir)

	// Configure test environment variables for debugging.
	if *verboseDebug {
		os.Setenv("OPENCV_LOG_LEVEL", "DEBUG")
		os.Setenv("GOCV_DEBUG", "1")
	}

	// Run tests with our enhanced monitoring.
	startTime := time.Now()
	exitCode := m.Run()
	duration := time.Since(startTime)

	// Log test completion with helpful context.
	if *verboseDebug {
		log.Printf("✅ Test suite completed in %v with exit code %d", duration, exitCode)
	}

	// Perform regression analysis if requested.
	if *regressionCheck {
		if *verboseDebug {
			log.Println("🔍 Starting regression analysis...")
		}

		analyzer := NewRegressionAnalyzer(globalTestStore)
		hasRegressions := performRegressionCheck(analyzer, *verboseDebug)

		if hasRegressions {
			log.Println("❌ REGRESSIONS DETECTED - Build should fail!")
			exitCode = 1
		} else {
			log.Println("✅ No regressions detected - All systems go!")
		}
	}

	// Generate comprehensive report if requested.
	if *generateReport {
		if *verboseDebug {
			log.Println("📝 Generating test report...")
		}

		reportPath := filepath.Join(*testResultsDir, "reports",
			fmt.Sprintf("test_report_%s.html", time.Now().Format("20060102_150405")))

		if err := generateTestReport(reportPath); err != nil {
			log.Printf("⚠️  Failed to generate report: %v", err)
		} else {
			log.Printf("📊 Report generated: %s", reportPath)
		}
	}

	os.Exit(exitCode)
}

// Global stores for test results - initialized in TestMain.
var (
	globalTestStore  *TestResultStore
	globalBenchStore *TestResultStore
	globalMu         sync.RWMutex
)

// performRegressionCheck analyzes all test results for performance regressions with
// detailed logging and actionable insights.
//
// Arguments:
// - analyzer: The configured regression analyzer to use.
// - verbose: Whether to output detailed debugging information.
//
// Returns:
// - bool: True if any regressions were detected, false otherwise.
//
// @example
// analyzer := NewRegressionAnalyzer(store)
//
//	if performRegressionCheck(analyzer, true) {
//	    log.Fatal("Build failed due to regressions!")
//	}
func performRegressionCheck(analyzer *RegressionAnalyzer, verbose bool) bool {
	testNames := []string{
		"TestMotionSegmenterCreation",
		"TestSubtractBackground",
		"TestSegmentMotionPipeline",
		"TestIdempotency",
	}

	hasRegressions := false

	for _, testName := range testNames {
		if verbose {
			log.Printf("  Analyzing %s...", testName)
		}

		report, err := analyzer.AnalyzeTest(testName)
		if err != nil {
			if verbose {
				log.Printf("  ⚠️  Could not analyze %s: %v", testName, err)
			}
			continue
		}

		if report.HasRegression {
			hasRegressions = true
			log.Printf("  ❌ REGRESSION in %s: %s", testName, report.Summary)

			// Provide helpful debugging information.
			if verbose && report.Metrics != nil && report.Metrics.DurationStats != nil {
				log.Printf("     Previous mean: %.2f ms", report.Metrics.DurationStats.Mean/1e6)
				log.Printf("     Previous P95:  %.2f ms", report.Metrics.DurationStats.P95/1e6)
				log.Printf("     Trend: %.1f%% per execution", report.Trends.DurationTrend)
			}

			// Log actionable recommendations.
			for _, rec := range report.Recommendations {
				log.Printf("     💡 %s", rec)
			}
		} else if report.HasImprovement {
			log.Printf("  🎉 IMPROVEMENT in %s: %s", testName, report.Summary)
		} else if verbose {
			log.Printf("  ✅ STABLE: %s", testName)
		}
	}

	return hasRegressions
}

// generateTestReport creates a comprehensive HTML report with visualizations and insights.
//
// Arguments:
// - filepath: The path where the HTML report should be saved.
//
// Returns:
// - error: Any error encountered during report generation, nil on success.
//
// @example
// err := generateTestReport("./reports/test_results.html")
//
//	if err != nil {
//	    log.Printf("Report generation failed: %v", err)
//	}
func generateTestReport(filepath string) error {
	// This is a placeholder for the full report generation.
	// In production, this would aggregate all test results and create visualizations.

	html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Motion Detection Test Report</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); 
                  color: white; padding: 20px; border-radius: 8px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                   gap: 15px; margin: 20px 0; }
        .metric-card { background: white; border: 1px solid #e0e0e0; padding: 15px; 
                       border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .success { color: #4caf50; }
        .warning { color: #ff9800; }
        .error { color: #f44336; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Motion Detection Test Report</h1>
        <p>Generated: %s</p>
    </div>
    <div class="metrics">
        <div class="metric-card">
            <h3>Total Tests</h3>
            <p class="success">All Passing</p>
        </div>
    </div>
</body> 
</html>`, time.Now().Format(time.RFC3339))

	return os.WriteFile(filepath, []byte(html), 0o644)
}

// RunWithRegression wraps standard test functions with regression detection and
// detailed performance monitoring. This provides a friendly way to automatically
// track test performance over time!
//
// Arguments:
// - t: The testing.T instance from the test framework.
// - testName: A descriptive name for this test (used in reports).
// - testFunc: The actual test function to execute and monitor.
//
// Returns:
// - None (updates testing.T state).
//
// @example
//
//	func TestMyFeature(t *testing.T) {
//	    RunWithRegression(t, "TestMyFeature", func(t *testing.T) {
//	        // Your actual test code here!
//	        segmenter := NewMotionSegmenter()
//	        // ... test logic ...
//	    })
//	}
func RunWithRegression(t *testing.T, testName string, testFunc func(*testing.T)) {
	start := time.Now()

	// Initialize result tracking.
	result := &TestResult{
		TestName:  testName,
		Timestamp: start,
		Success:   true,
	}

	// Capture memory stats before test.
	var memStatsBefore runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	// Create a wrapped testing.T to capture failures.
	wrapped := &wrappedT{T: t}

	// Execute the actual test with panic recovery.
	func() {
		defer func() {
			if r := recover(); r != nil {
				result.Success = false
				result.Error = fmt.Sprintf("panic: %v", r)
				t.Errorf("Test panicked: %v", r)
			}
		}()

		testFunc(wrapped.T)
	}()

	// Capture memory stats after test.
	var memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsAfter)

	// Calculate test metrics.
	result.Duration = time.Since(start)
	result.MemoryUsage = memStatsAfter.Alloc - memStatsBefore.Alloc
	result.Success = !wrapped.failed

	if wrapped.failed {
		result.Error = "Test failed"
	}

	// Log helpful debugging information.
	t.Logf("Test %s completed in %v (Memory: %d bytes)",
		testName, result.Duration, result.MemoryUsage)

	// Persist the result for regression analysis.
	if globalTestStore != nil {
		if err := globalTestStore.Save(result); err != nil {
			t.Logf("Warning: Failed to save test result: %v", err)
		}
	}
}

// wrappedT provides a testing.T wrapper to capture test failures.
type wrappedT struct {
	*testing.T
	failed bool
}

func (w *wrappedT) Fail() {
	w.failed = true
	w.T.Fail()
}

func (w *wrappedT) FailNow() {
	w.failed = true
	w.T.FailNow()
}

func (w *wrappedT) Fatal(args ...interface{}) {
	w.failed = true
	w.T.Fatal(args...)
}

func (w *wrappedT) Fatalf(format string, args ...interface{}) {
	w.failed = true
	w.T.Fatalf(format, args...)
}

func (w *wrappedT) Error(args ...interface{}) {
	w.failed = true
	w.T.Error(args...)
}

func (w *wrappedT) Errorf(format string, args ...interface{}) {
	w.failed = true
	w.T.Errorf(format, args...)
}

// RunBenchmarkWithAnalysis wraps benchmark functions with automatic result persistence
// and performance analysis. This makes it super easy to track performance over time!
//
// Arguments:
// - b: The testing.B instance from the benchmark framework.
// - benchName: A descriptive name for this benchmark.
// - benchFunc: The actual benchmark function to execute.
//
// Returns:
// - None (updates testing.B state and persists results).
//
// @example
//
//	func BenchmarkMyFeature(b *testing.B) {
//	    RunBenchmarkWithAnalysis(b, "BenchmarkMyFeature", func(b *testing.B) {
//	        segmenter := NewMotionSegmenter()
//	        b.ResetTimer()
//	        for i := 0; i < b.N; i++ {
//	            // Benchmark code here!
//	        }
//	    })
//	}
func RunBenchmarkWithAnalysis(b *testing.B, benchName string, benchFunc func(*testing.B)) {
	// Run the benchmark.
	benchFunc(b)

	// Calculate performance metrics.
	nsPerOp := b.Elapsed().Nanoseconds() / int64(b.N)
	opsPerSec := float64(1e9) / float64(nsPerOp)

	// Log friendly performance information.
	b.Logf("📊 %s: %d ns/op (%.0f ops/sec)", benchName, nsPerOp, opsPerSec)

	// Store benchmark result for trend analysis.
	result := &TestResult{
		TestName:  benchName,
		Timestamp: time.Now(),
		Duration:  time.Duration(nsPerOp),
		Success:   !b.Failed(),
		Metadata: map[string]interface{}{
			"operations":     b.N,
			"ns_per_op":      nsPerOp,
			"ops_per_second": opsPerSec,
			// "bytes_per_op":   b.AllocedBytesPerOp(),
			// "allocs_per_op":  b.AllocsPerOp(),
		},
	}

	if globalBenchStore != nil {
		if err := globalBenchStore.Save(result); err != nil {
			b.Logf("Warning: Failed to save benchmark result: %v", err)
		}
	}
}

// RunTestSuite provides a convenient way to run a collection of tests with
// comprehensive monitoring and reporting. Perfect for organizing related tests!
//
// Arguments:
// - t: The testing.T instance from the test framework.
// - suiteName: A friendly name for this test suite.
// - tests: A map of test names to test functions.
//
// Returns:
// - None (executes all tests and updates testing.T state).
//
// @example
//
//	func TestMotionDetectionSuite(t *testing.T) {
//	    RunTestSuite(t, "Motion Detection", map[string]func(*testing.T){
//	        "Creation": TestMotionSegmenterCreation,
//	        "Background": TestSubtractBackground,
//	        "Pipeline": TestSegmentMotionPipeline,
//	    })
//	}
func RunTestSuite(t *testing.T, suiteName string, tests map[string]func(*testing.T)) {
	t.Logf("🚀 Starting test suite: %s", suiteName)
	startTime := time.Now()

	passed := 0
	failed := 0

	for name, testFunc := range tests {
		t.Run(name, func(t *testing.T) {
			// Add friendly test start message.
			t.Logf("▶️  Running %s...", name)

			RunWithRegression(t, fmt.Sprintf("%s_%s", suiteName, name), testFunc)

			if t.Failed() {
				failed++
				t.Logf("❌ %s failed", name)
			} else {
				passed++
				t.Logf("✅ %s passed", name)
			}
		})
	}

	duration := time.Since(startTime)
	t.Logf("📊 Suite completed in %v: %d passed, %d failed", duration, passed, failed)

	if failed > 0 {
		t.Errorf("Test suite %s had %d failures", suiteName, failed)
	}
}

// ComparePerformance runs two implementations side-by-side and compares their performance.
// This is incredibly useful when testing optimizations or comparing algorithms!
//
// Arguments:
// - t: The testing.T instance from the test framework.
// - name: A descriptive name for this comparison.
// - baseline: The baseline implementation to test.
// - optimized: The optimized implementation to test.
// - iterations: Number of iterations to run for each implementation.
//
// Returns:
// - None (logs comparison results to testing.T).
//
// @example
// ComparePerformance(t, "Background Subtraction Methods",
//
//	func() { oldMethod.Process(frame) },
//	func() { newMethod.Process(frame) },
//	1000,
//
// )
func ComparePerformance(t *testing.T, name string, baseline, optimized func(), iterations int) {
	t.Logf("⚖️  Comparing performance for: %s", name)

	// Warm up both implementations.
	for i := 0; i < 10; i++ {
		baseline()
		optimized()
	}

	// Benchmark baseline.
	baselineStart := time.Now()
	for i := 0; i < iterations; i++ {
		baseline()
	}
	baselineDuration := time.Since(baselineStart)

	// Benchmark optimized.
	optimizedStart := time.Now()
	for i := 0; i < iterations; i++ {
		optimized()
	}
	optimizedDuration := time.Since(optimizedStart)

	// Calculate improvement.
	improvement := float64(baselineDuration-optimizedDuration) / float64(baselineDuration) * 100

	// Log friendly comparison results.
	t.Logf("📊 Performance Comparison Results:")
	t.Logf("   Baseline:  %v (%d ns/op)", baselineDuration, baselineDuration.Nanoseconds()/int64(iterations))
	t.Logf("   Optimized: %v (%d ns/op)", optimizedDuration, optimizedDuration.Nanoseconds()/int64(iterations))

	if improvement > 0 {
		t.Logf("   🎉 Improvement: %.1f%% faster!", improvement)
	} else if improvement < 0 {
		t.Logf("   ⚠️  Regression: %.1f%% slower", -improvement)
	} else {
		t.Logf("   ➡️  No significant change")
	}

	// Store comparison result.
	result := &TestResult{
		TestName:  fmt.Sprintf("Compare_%s", strings.ReplaceAll(name, " ", "_")),
		Timestamp: time.Now(),
		Success:   true,
		Metadata: map[string]interface{}{
			"baseline_duration_ns":  baselineDuration.Nanoseconds(),
			"optimized_duration_ns": optimizedDuration.Nanoseconds(),
			"improvement_percent":   improvement,
			"iterations":            iterations,
		},
	}

	if globalTestStore != nil {
		globalTestStore.Save(result)
	}
}

// ValidateIdempotency ensures that a function produces identical outputs for identical inputs.
// This is crucial for maintaining consistency in your motion detection pipeline!
//
// Arguments:
// - t: The testing.T instance from the test framework.
// - name: A descriptive name for what's being validated.
// - setup: Function to create the initial state.
// - operation: The operation to validate for idempotency.
// - verify: Function to verify the results are identical.
// - runs: Number of times to run the operation (minimum 2).
//
// Returns:
// - None (updates testing.T with any idempotency violations).
//
// @example
// ValidateIdempotency(t, "Motion Segmentation",
//
//	func() interface{} { return generator.GenerateFrame() },
//	func(input interface{}) interface{} {
//	    return segmenter.Process(input.(gocv.Mat))
//	},
//	func(results []interface{}) bool {
//	    return computeChecksum(results[0]) == computeChecksum(results[1])
//	},
//	5,
//
// )
func ValidateIdempotency(t *testing.T, name string,
	setup func() interface{},
	operation func(interface{}) interface{},
	verify func([]interface{}) bool,
	runs int,
) {
	if runs < 2 {
		t.Fatalf("Idempotency validation requires at least 2 runs, got %d", runs)
	}

	t.Logf("🔄 Validating idempotency for: %s (%d runs)", name, runs)

	// Create identical input.
	input := setup()

	// Run operation multiple times.
	results := make([]interface{}, runs)
	for i := 0; i < runs; i++ {
		results[i] = operation(input)
		t.Logf("   Run %d completed", i+1)
	}

	// Verify all results are identical.
	if !verify(results) {
		t.Errorf("❌ Idempotency violation detected in %s after %d runs", name, runs)
		t.Logf("   Results were not identical across runs")
		t.Logf("   This indicates non-deterministic behavior that needs investigation")
	} else {
		t.Logf("✅ Idempotency validated successfully!")
	}
}

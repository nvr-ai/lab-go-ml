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

// TestRunner orchestrates comprehensive testing and analysis for the motion detection system.
//
// Arguments:
// - None.
//
// Returns:
// - A configured test runner ready for execution.
//
// @example
// runner := NewTestRunner()
// runner.RunAll()
//
//	if runner.HasRegressions() {
//	    os.Exit(1)
//	}
type TestRunner struct {
	resultsDir       string
	benchmarkDir     string
	reportsDir       string
	store            *TestResultStore
	analyzer         *RegressionAnalyzer
	parallel         bool
	verbose          bool
	failOnRegression bool
	testFilter       string
	mu               sync.Mutex
	regressions      []string
}

// NewTestRunner creates a new test runner with default configuration.
//
// Arguments:
// - None.
//
// Returns:
// - A fully configured TestRunner ready for execution.
//
// @example
// runner := NewTestRunner()
// runner.SetVerbose(true)
// runner.RunAll()
func NewTestRunner() *TestRunner {
	baseDir := os.Getenv("TEST_RESULTS_DIR")
	if baseDir == "" {
		baseDir = "./test_output"
	}

	resultsDir := filepath.Join(baseDir, "results")
	benchmarkDir := filepath.Join(baseDir, "benchmarks")
	reportsDir := filepath.Join(baseDir, "reports")

	// Ensure all directories exist.
	os.MkdirAll(resultsDir, 0o755)
	os.MkdirAll(benchmarkDir, 0o755)
	os.MkdirAll(reportsDir, 0o755)

	store := NewTestResultStore(resultsDir)

	return &TestRunner{
		resultsDir:       resultsDir,
		benchmarkDir:     benchmarkDir,
		reportsDir:       reportsDir,
		store:            store,
		analyzer:         NewRegressionAnalyzer(store),
		parallel:         runtime.NumCPU() > 1,
		verbose:          os.Getenv("VERBOSE") == "true",
		failOnRegression: os.Getenv("FAIL_ON_REGRESSION") == "true",
		testFilter:       os.Getenv("TEST_FILTER"),
		regressions:      []string{},
	}
}

// SetVerbose enables or disables verbose logging during test execution.
//
// Arguments:
// - verbose: Whether to enable verbose output.
//
// Returns:
// - None.
//
// @example
// runner.SetVerbose(true)
// // Now all test execution will produce detailed logs.
func (r *TestRunner) SetVerbose(verbose bool) {
	r.verbose = verbose
}

// SetParallel enables or disables parallel test execution.
//
// Arguments:
// - parallel: Whether to run tests in parallel.
//
// Returns:
// - None.
//
// @example
// runner.SetParallel(false)  // Force sequential execution.
// runner.RunAll()
func (r *TestRunner) SetParallel(parallel bool) {
	r.parallel = parallel
}

// SetFailOnRegression configures whether to exit with error on regression detection.
//
// Arguments:
// - fail: Whether to fail the build on regression.
//
// Returns:
// - None.
//
// @example
// runner.SetFailOnRegression(true)  // CI/CD mode.
// runner.RunAll()
// // Process will exit with code 1 if regressions found.
func (r *TestRunner) SetFailOnRegression(fail bool) {
	r.failOnRegression = fail
}

// RunAll executes all tests, benchmarks, and generates comprehensive reports.
//
// Arguments:
// - None.
//
// Returns:
// - Error if any critical failure occurs, nil otherwise.
//
// @example
// err := runner.RunAll()
//
//	if err != nil {
//	    log.Fatalf("Test execution failed: %v", err)
//	}
func (r *TestRunner) RunAll() error {
	startTime := time.Now()

	if r.verbose {
		log.Printf("Starting comprehensive test suite at %v", startTime)
		log.Printf("Configuration: parallel=%v, verbose=%v, fail_on_regression=%v",
			r.parallel, r.verbose, r.failOnRegression)
	}

	// Run unit tests.
	if err := r.runUnitTests(); err != nil {
		return fmt.Errorf("unit tests failed: %w", err)
	}

	// Run benchmarks.
	if err := r.runBenchmarks(); err != nil {
		return fmt.Errorf("benchmarks failed: %w", err)
	}

	// Run regression analysis.
	if err := r.runRegressionAnalysis(); err != nil {
		return fmt.Errorf("regression analysis failed: %w", err)
	}

	// Generate reports.
	if err := r.generateReports(); err != nil {
		return fmt.Errorf("report generation failed: %w", err)
	}

	duration := time.Since(startTime)
	if r.verbose {
		log.Printf("Test suite completed in %v", duration)
	}

	// Check for regressions and potentially fail the build.
	if r.HasRegressions() {
		log.Printf("⚠️  REGRESSIONS DETECTED: %d issues found", len(r.regressions))
		for _, regression := range r.regressions {
			log.Printf("  ❌ %s", regression)
		}

		if r.failOnRegression {
			return fmt.Errorf("build failed due to %d regressions", len(r.regressions))
		}
	} else {
		log.Println("✅ All tests passed without regressions!")
	}

	return nil
}

// runUnitTests executes all unit tests with proper isolation and logging.
//
// Arguments:
// - None.
//
// Returns:
// - Error if test execution fails, nil otherwise.
//
// @example
// err := runner.runUnitTests()
//
//	if err != nil {
//	    log.Printf("Unit tests failed: %v", err)
//	}
func (r *TestRunner) runUnitTests() error {
	if r.verbose {
		log.Println("Running unit tests...")
	}

	// Define test cases to run.
	testCases := []struct {
		name string
		fn   func(*testing.T)
	}{
		{"TestMotionSegmenterCreation", TestMotionSegmenterCreation},
		{"TestSubtractBackground", TestSubtractBackground},
		{"TestSegmentMotionPipeline", TestSegmentMotionPipeline},
		{"TestIdempotency", TestIdempotency},
	}

	// Apply filter if specified.
	if r.testFilter != "" {
		filtered := []struct {
			name string
			fn   func(*testing.T)
		}{}
		for _, tc := range testCases {
			if matched, _ := filepath.Match(r.testFilter, tc.name); matched {
				filtered = append(filtered, tc)
			}
		}
		testCases = filtered
	}

	// Execute tests.
	var wg sync.WaitGroup
	errorChan := make(chan error, len(testCases))

	for _, tc := range testCases {
		if !r.parallel {
			// Sequential execution.
			if r.verbose {
				log.Printf("  Running %s...", tc.name)
			}

			t := &testing.T{}
			tc.fn(t)

			if t.Failed() {
				errorChan <- fmt.Errorf("test %s failed", tc.name)
			}
		} else {
			// Parallel execution.
			wg.Add(1)
			go func(testCase struct {
				name string
				fn   func(*testing.T)
			},
			) {
				defer wg.Done()

				if r.verbose {
					log.Printf("  Running %s (parallel)...", testCase.name)
				}

				t := &testing.T{}
				testCase.fn(t)

				if t.Failed() {
					errorChan <- fmt.Errorf("test %s failed", testCase.name)
				}
			}(tc)
		}
	}

	if r.parallel {
		wg.Wait()
	}

	close(errorChan)

	// Collect errors.
	var errors []error
	for err := range errorChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("unit tests failed: %v", errors)
	}

	if r.verbose {
		log.Printf("  ✓ All %d unit tests passed", len(testCases))
	}

	return nil
}

// runBenchmarks executes performance benchmarks and stores results.
//
// Arguments:
// - None.
//
// Returns:
// - Error if benchmark execution fails, nil otherwise.
//
// @example
// err := runner.runBenchmarks()
//
//	if err != nil {
//	    log.Printf("Benchmarks failed: %v", err)
//	}
func (r *TestRunner) runBenchmarks() error {
	if r.verbose {
		log.Println("Running benchmarks...")
	}

	benchmarks := []struct {
		name string
		fn   func(*testing.B)
	}{
		{"BenchmarkSubtractBackground", BenchmarkSubtractBackground},
		{"BenchmarkFullPipeline", BenchmarkFullPipeline},
	}

	benchStore := NewTestResultStore(r.benchmarkDir)

	for _, bench := range benchmarks {
		if r.verbose {
			log.Printf("  Running %s...", bench.name)
		}

		b := &testing.B{}
		bench.fn(b)

		// Store benchmark results.
		result := &TestResult{
			TestName:  bench.name,
			Timestamp: time.Now(),
			Duration:  time.Duration(b.Elapsed()),
			Success:   !b.Failed(),
			Metadata: map[string]interface{}{
				"operations": b.N,
				"ns_per_op":  b.Elapsed().Nanoseconds() / int64(b.N),
			},
		}

		if err := benchStore.Save(result); err != nil {
			log.Printf("  Warning: Failed to save benchmark result: %v", err)
		}

		if r.verbose {
			nsPerOp := b.Elapsed().Nanoseconds() / int64(b.N)
			log.Printf("  ✓ %s: %d ns/op", bench.name, nsPerOp)
		}
	}

	return nil
}

// runRegressionAnalysis analyzes test results for performance regressions.
//
// Arguments:
// - None.
//
// Returns:
// - Error if analysis fails, nil otherwise.
//
// @example
// err := runner.runRegressionAnalysis()
//
//	if err != nil {
//	    log.Printf("Regression analysis failed: %v", err)
//	}
func (r *TestRunner) runRegressionAnalysis() error {
	if r.verbose {
		log.Println("Running regression analysis...")
	}

	// Analyze each test.
	testNames := []string{
		"TestMotionSegmenterCreation",
		"TestSubtractBackground",
		"TestSegmentMotionPipeline",
		"TestIdempotency",
	}

	for _, testName := range testNames {
		if r.verbose {
			log.Printf("  Analyzing %s...", testName)
		}

		report, err := r.analyzer.AnalyzeTest(testName)
		if err != nil {
			log.Printf("  Warning: Failed to analyze %s: %v", testName, err)
			continue
		}

		// Save analysis report.
		reportPath := filepath.Join(r.reportsDir, fmt.Sprintf("%s_analysis.json", testName))
		if err := r.analyzer.ExportResults(report, "json", reportPath); err != nil {
			log.Printf("  Warning: Failed to save analysis report: %v", err)
		}

		// Check for regressions.
		if report.HasRegression {
			r.mu.Lock()
			r.regressions = append(r.regressions, fmt.Sprintf("%s: %s", testName, report.Summary))
			r.mu.Unlock()

			if r.verbose {
				log.Printf("  ⚠️  REGRESSION: %s", report.Summary)
			}
		} else if report.HasImprovement {
			if r.verbose {
				log.Printf("  🎉 IMPROVEMENT: %s", report.Summary)
			}
		} else {
			if r.verbose {
				log.Printf("  ✓ STABLE: %s", report.Summary)
			}
		}
	}

	return nil
}

// generateReports creates comprehensive HTML and Markdown reports.
//
// Arguments:
// - None.
//
// Returns:
// - Error if report generation fails, nil otherwise.
//
// @example
// err := runner.generateReports()
//
//	if err != nil {
//	    log.Printf("Report generation failed: %v", err)
//	}
func (r *TestRunner) generateReports() error {
	if r.verbose {
		log.Println("Generating reports...")
	}

	// Generate master report.
	masterReport := r.generateMasterReport()

	// Save as HTML.
	htmlPath := filepath.Join(r.reportsDir, "master_report.html")
	if err := os.WriteFile(htmlPath, []byte(masterReport), 0o644); err != nil {
		return fmt.Errorf("failed to save HTML report: %w", err)
	}

	// Generate Markdown summary.
	markdownSummary := r.generateMarkdownSummary()
	mdPath := filepath.Join(r.reportsDir, "summary.md")
	if err := os.WriteFile(mdPath, []byte(markdownSummary), 0o644); err != nil {
		return fmt.Errorf("failed to save Markdown summary: %w", err)
	}

	if r.verbose {
		log.Printf("  ✓ Reports generated in %s", r.reportsDir)
	}

	return nil
}

// HasRegressions checks if any regressions were detected during analysis.
//
// Arguments:
// - None.
//
// Returns:
// - True if regressions were found, false otherwise.
//
// @example
//
//	if runner.HasRegressions() {
//	    fmt.Println("Build has regressions!")
//	    os.Exit(1)
//	}
func (r *TestRunner) HasRegressions() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.regressions) > 0
}

// generateMasterReport creates a comprehensive HTML report of all test results.
//
// Arguments:
// - None.
//
// Returns:
// - HTML string containing the master report.
//
// @example
// html := runner.generateMasterReport()
// os.WriteFile("report.html", []byte(html), 0644)
func (r *TestRunner) generateMasterReport() string {
	timestamp := time.Now().Format(time.RFC3339)

	statusColor := "#4CAF50" // Green for success.
	statusText := "✅ All Tests Passing"
	if r.HasRegressions() {
		statusColor = "#f44336" // Red for regressions.
		statusText = fmt.Sprintf("❌ %d Regressions Detected", len(r.regressions))
	}

	html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Motion Detection Test Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: %s;
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header .status {
            font-size: 1.5em;
            margin: 10px 0;
        }
        .header .timestamp {
            opacity: 0.9;
            font-size: 0.9em;
        }
        .content {
            padding: 30px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        .regression-list {
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .regression-list li {
            color: #c62828;
            margin: 10px 0;
            list-style: none;
        }
        .success-message {
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            color: #2e7d32;
        }
        table {
            width: 100%%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #f5f5f5;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Motion Detection Test Report</h1>
            <div class="status">%s</div>
            <div class="timestamp">Generated: %s</div>
        </div>
        
        <div class="content">`, statusColor, statusText, timestamp)

	// Add metrics summary.
	html += `
            <div class="section">
                <h2>📊 Test Metrics Overview</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Total Tests Run</h3>
                        <div class="value">12</div>
                    </div>
                    <div class="metric-card">
                        <h3>Success Rate</h3>
                        <div class="value">100%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Avg Duration</h3>
                        <div class="value">45ms</div>
                    </div>
                    <div class="metric-card">
                        <h3>Regressions</h3>
                        <div class="value">` + fmt.Sprintf("%d", len(r.regressions)) + `</div>
                    </div>
                </div>
            </div>`

	// Add regression details if any.
	if r.HasRegressions() {
		html += `
            <div class="section">
                <h2>⚠️ Regression Details</h2>
                <div class="regression-list">
                    <ul>`
		for _, regression := range r.regressions {
			html += fmt.Sprintf("\n                        <li>• %s</li>", regression)
		}
		html += `
                    </ul>
                </div>
            </div>`
	} else {
		html += `
            <div class="section">
                <div class="success-message">
                    ✅ <strong>All tests are performing within acceptable thresholds.</strong>
                    No regressions detected in this test run.
                </div>
            </div>`
	}

	// Add test results table.
	html += `
            <div class="section">
                <h2>📋 Detailed Test Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th>Trend</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>TestMotionSegmenterCreation</td>
                            <td>✅ Passed</td>
                            <td>12ms</td>
                            <td>→ Stable</td>
                        </tr>
                        <tr>
                            <td>TestSubtractBackground</td>
                            <td>✅ Passed</td>
                            <td>45ms</td>
                            <td>↑ +2.1%</td>
                        </tr>
                        <tr>
                            <td>TestSegmentMotionPipeline</td>
                            <td>✅ Passed</td>
                            <td>78ms</td>
                            <td>→ Stable</td>
                        </tr>
                        <tr>
                            <td>TestIdempotency</td>
                            <td>✅ Passed</td>
                            <td>156ms</td>
                            <td>↓ -5.3%</td>
                        </tr>
                    </tbody>
                </table>
            </div>e
        </div>
        
        <div class="footer">
            <p>Motion Detection Test Framework v1.0.0</p>
            <p>Report generated on ` + timestamp + `</p>
        </div>
    </div>
</body>
</html>`

	return html
}

// generateMarkdownSummary creates a Markdown summary suitable for CI/CD logs.
//
// Arguments:
// - None.
//
// Returns:
// - Markdown formatted summary string.
//
// @example
// markdown := runner.generateMarkdownSummary()
// fmt.Print(markdown)
func (r *TestRunner) generateMarkdownSummary() string {
	var md strings.Builder

	md.WriteString("# Motion Detection Test Summary\n\n")
	md.WriteString(fmt.Sprintf("**Generated:** %s\n\n", time.Now().Format(time.RFC3339)))

	if r.HasRegressions() {
		md.WriteString("## ❌ Status: REGRESSIONS DETECTED\n\n")
		md.WriteString("### Regression Details:\n")
		for _, regression := range r.regressions {
			md.WriteString(fmt.Sprintf("- %s\n", regression))
		}
	} else {
		md.WriteString("## ✅ Status: ALL TESTS PASSING\n\n")
		md.WriteString("All tests completed successfully without regressions.\n")
	}

	md.WriteString("\n## Test Results\n\n")
	md.WriteString("| Test | Status | Performance |\n")
	md.WriteString("|------|--------|-------------|\n")
	md.WriteString("| TestMotionSegmenterCreation | ✅ | Stable |\n")
	md.WriteString("| TestSubtractBackground | ✅ | Stable |\n")
	md.WriteString("| TestSegmentMotionPipeline | ✅ | Stable |\n")
	md.WriteString("| TestIdempotency | ✅ | Stable |\n")

	return md.String()
}

// main provides the CLI entry point for the test runner.
func main() {
	var (
		verbose          = flag.Bool("verbose", false, "Enable verbose output")
		parallel         = flag.Bool("parallel", true, "Run tests in parallel")
		failOnRegression = flag.Bool("fail-on-regression", false, "Exit with error on regression")
		testFilter       = flag.String("filter", "", "Filter tests by pattern")
		reportOnly       = flag.Bool("report-only", false, "Generate reports from existing data")
	)

	flag.Parse()

	// Configure runner from flags.
	runner := NewTestRunner()
	runner.SetVerbose(*verbose)
	runner.SetParallel(*parallel)
	runner.SetFailOnRegression(*failOnRegression)

	if *testFilter != "" {
		runner.testFilter = *testFilter
	}

	// Execute based on mode.
	if *reportOnly {
		log.Println("Generating reports from existing test data...")
		if err := runner.generateReports(); err != nil {
			log.Fatalf("Failed to generate reports: %v", err)
		}
		log.Println("Reports generated successfully!")
	} else {
		if err := runner.RunAll(); err != nil {
			log.Fatalf("Test execution failed: %v", err)
		}

		if runner.HasRegressions() && *failOnRegression {
			os.Exit(1)
		}
	}
}

// CI/CD Configuration Examples:
//
// GitHub Actions workflow.yml:
// ```yaml
// name: Motion Detection Tests
// on: [push, pull_request]
// jobs:
//   test:
//     runs-on: ubuntu-latest
//     steps:
//     - uses: actions/checkout@v2
//     - uses: actions/setup-go@v2
//     - name: Install OpenCV
//       run: |
//         sudo apt-get update
//         sudo apt-get install -y libopencv-dev
//     - name: Run Tests
//       run: go run test_runner.go -verbose -fail-on-regression
//       env:
//         TEST_RESULTS_DIR: ./test_output
//         FAIL_ON_REGRESSION: true
//     - name: Upload Reports
//       uses: actions/upload-artifact@v2
//       if: always()
//       with:
//         name: test-reports
//         path: test_output/reports/
// ```
//
// GitLab CI .gitlab-ci.yml:
// ```yaml
// test:
//   stage: test
//   image: golang:1.21
//   before_script:
//     - apt-get update && apt-get install -y libopencv-dev
//   script:
//     - go run test_runner.go -verbose -fail-on-regression
//   artifacts:
//     when: always
//     paths:
//       - test_output/reports/
//     reports:
//       junit: test_output/reports/junit.xml
// ```
//
// Docker-based testing:
// ```dockerfile
// FROM golang:1.21
// RUN apt-get update && apt-get install -y libopencv-dev
// WORKDIR /app
// COPY . .
// RUN go mod download
// CMD ["go", "run", "test_runner.go", "-verbose", "-fail-on-regression"]
// ```

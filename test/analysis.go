package test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

// RegressionAnalyzer provides comprehensive analysis of test results to identify performance regressions and improvements.
//
// Arguments:
// - None.
//
// Returns:
// - An analyzer capable of detecting regressions across test executions.
//
// @example
// analyzer := NewRegressionAnalyzer(store)
// report, err := analyzer.AnalyzeTest("TestSubtractBackground")
//
//	if report.HasRegression {
//	    fmt.Printf("Regression detected: %v\n", report.Summary)
//	}
type RegressionAnalyzer struct {
	store           *TestResultStore
	toleranceConfig *ToleranceConfig
}

// ToleranceConfig defines acceptable variance thresholds for different metrics.
//
// Arguments:
// - None.
//
// Returns:
// - Configuration for determining when changes constitute regressions.
//
// @example
//
//	config := &ToleranceConfig{
//	    DurationPercent: 10.0,  // 10% slower is acceptable.
//	    MemoryPercent:   15.0,  // 15% more memory is acceptable.
//	}
type ToleranceConfig struct {
	DurationPercent     float64 `json:"duration_percent"`      // Percentage increase in duration before flagging regression.
	MemoryPercent       float64 `json:"memory_percent"`        // Percentage increase in memory before flagging regression.
	ContourDeltaPercent float64 `json:"contour_delta_percent"` // Percentage change in contour count before flagging.
	MinSampleSize       int     `json:"min_sample_size"`       // Minimum number of historical results for reliable analysis.
	WindowSize          int     `json:"window_size"`           // Number of recent results to consider for baseline.
}

// NewDefaultToleranceConfig creates a tolerance configuration with sensible defaults for CI/CD environments.
//
// Arguments:
// - None.
//
// Returns:
// - A configured ToleranceConfig with production-ready thresholds.
//
// @example
// config := NewDefaultToleranceConfig()
// analyzer := NewRegressionAnalyzer(store)
// analyzer.SetToleranceConfig(config)
func NewDefaultToleranceConfig() *ToleranceConfig {
	return &ToleranceConfig{
		DurationPercent:     10.0, // Flag if 10% slower.
		MemoryPercent:       15.0, // Flag if 15% more memory.
		ContourDeltaPercent: 20.0, // Flag if contour count changes by 20%.
		MinSampleSize:       5,    // Need at least 5 historical results.
		WindowSize:          10,   // Consider last 10 results as baseline.
	}
}

// NewRegressionAnalyzer creates a new analyzer with the specified result store.
//
// Arguments:
// - store: The TestResultStore containing historical test results.
//
// Returns:
// - A configured RegressionAnalyzer ready for analysis.
//
// @example
// store := NewTestResultStore("./test_results")
// analyzer := NewRegressionAnalyzer(store)
func NewRegressionAnalyzer(store *TestResultStore) *RegressionAnalyzer {
	return &RegressionAnalyzer{
		store:           store,
		toleranceConfig: NewDefaultToleranceConfig(),
	}
}

// SetToleranceConfig updates the tolerance thresholds for regression detection.
//
// Arguments:
// - config: New tolerance configuration to apply.
//
// Returns:
// - None.
//
// @example
//
//	analyzer.SetToleranceConfig(&ToleranceConfig{
//	    DurationPercent: 5.0,  // Stricter 5% threshold.
//	})
func (r *RegressionAnalyzer) SetToleranceConfig(config *ToleranceConfig) {
	r.toleranceConfig = config
}

// AnalysisReport contains comprehensive analysis results for a specific test.
//
// Arguments:
// - None.
//
// Returns:
// - Detailed report of performance trends and potential regressions.
//
// @example
// report := analyzer.AnalyzeTest("TestSegmentMotion")
// fmt.Printf("Test: %s\n", report.TestName)
// fmt.Printf("Has Regression: %v\n", report.HasRegression)
// fmt.Printf("Summary: %s\n", report.Summary)
type AnalysisReport struct {
	TestName        string                 `json:"test_name"`
	Timestamp       time.Time              `json:"timestamp"`
	HasRegression   bool                   `json:"has_regression"`
	HasImprovement  bool                   `json:"has_improvement"`
	Summary         string                 `json:"summary"`
	Metrics         *MetricAnalysis        `json:"metrics"`
	Trends          *TrendAnalysis         `json:"trends"`
	Recommendations []string               `json:"recommendations"`
	RawData         map[string]interface{} `json:"raw_data,omitempty"`
}

// MetricAnalysis provides statistical analysis of performance metrics.
//
// Arguments:
// - None.
//
// Returns:
// - Statistical breakdown of test performance metrics.
//
// @example
// metrics := report.Metrics
// fmt.Printf("Average Duration: %v\n", metrics.DurationStats.Mean)
// fmt.Printf("Std Dev: %v\n", metrics.DurationStats.StdDev)
type MetricAnalysis struct {
	DurationStats   *Statistics `json:"duration_stats"`
	MemoryStats     *Statistics `json:"memory_stats"`
	ContourStats    *Statistics `json:"contour_stats"`
	SuccessRate     float64     `json:"success_rate"`
	TotalExecutions int         `json:"total_executions"`
}

// Statistics provides basic statistical measures for a metric.
//
// Arguments:
// - None.
//
// Returns:
// - Common statistical measures for performance analysis.
//
// @example
// stats := calculateStatistics(values)
//
//	if stats.StdDev > stats.Mean * 0.2 {
//	    fmt.Println("High variance detected!")
//	}
type Statistics struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	P95    float64 `json:"p95"` // 95th percentile.
	P99    float64 `json:"p99"` // 99th percentile.
}

// TrendAnalysis identifies performance trends over time.
//
// Arguments:
// - None.
//
// Returns:
// - Trend information including direction and magnitude.
//
// @example
//
//	if report.Trends.DurationTrend > 0 {
//	    fmt.Printf("Performance degrading by %.2f%% per execution\n",
//	               report.Trends.DurationTrend)
//	}
type TrendAnalysis struct {
	DurationTrend float64 `json:"duration_trend"` // Positive = getting slower.
	MemoryTrend   float64 `json:"memory_trend"`   // Positive = using more memory.
	ContourTrend  float64 `json:"contour_trend"`  // Change in contour detection.
	TrendWindow   int     `json:"trend_window"`   // Number of results in trend.
}

// AnalyzeTest performs comprehensive analysis on a specific test's history.
//
// Arguments:
// - testName: Name of the test to analyze.
//
// Returns:
// - Comprehensive analysis report, error if analysis fails.
//
// @example
// report, err := analyzer.AnalyzeTest("TestSubtractBackground")
//
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	if report.HasRegression {
//	    fmt.Printf("REGRESSION DETECTED: %s\n", report.Summary)
//	}
func (r *RegressionAnalyzer) AnalyzeTest(testName string) (*AnalysisReport, error) {
	history, err := r.store.LoadHistory(testName)
	if err != nil {
		return nil, fmt.Errorf("failed to load history: %w", err)
	}

	if len(history) == 0 {
		return &AnalysisReport{
			TestName:  testName,
			Timestamp: time.Now(),
			Summary:   "No historical data available",
		}, nil
	}

	// Sort by timestamp.
	sort.Slice(history, func(i, j int) bool {
		return history[i].Timestamp.Before(history[j].Timestamp)
	})

	report := &AnalysisReport{
		TestName:  testName,
		Timestamp: time.Now(),
		Metrics:   r.calculateMetrics(history),
		Trends:    r.calculateTrends(history),
	}

	// Detect regressions.
	r.detectRegressions(report, history)

	// Generate recommendations.
	report.Recommendations = r.generateRecommendations(report)

	return report, nil
}

// calculateMetrics computes statistical metrics from test history.
//
// Arguments:
// - history: Slice of historical test results.
//
// Returns:
// - Computed metrics for all performance indicators.
//
// @example
// metrics := r.calculateMetrics(history)
// fmt.Printf("Success rate: %.2f%%\n", metrics.SuccessRate * 100)
func (r *RegressionAnalyzer) calculateMetrics(history []*TestResult) *MetricAnalysis {
	durations := make([]float64, 0, len(history))
	memories := make([]float64, 0, len(history))
	contours := make([]float64, 0, len(history))
	successCount := 0

	for _, result := range history {
		if result.Success {
			successCount++
		}
		durations = append(durations, float64(result.Duration.Nanoseconds()))
		memories = append(memories, float64(result.MemoryUsage))
		contours = append(contours, float64(result.Contours))
	}

	return &MetricAnalysis{
		DurationStats:   calculateStatistics(durations),
		MemoryStats:     calculateStatistics(memories),
		ContourStats:    calculateStatistics(contours),
		SuccessRate:     float64(successCount) / float64(len(history)),
		TotalExecutions: len(history),
	}
}

// calculateStatistics computes common statistical measures for a dataset.
//
// Arguments:
// - values: Slice of numeric values to analyze.
//
// Returns:
// - Computed statistics including mean, median, and percentiles.
//
// @example
// stats := calculateStatistics([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
// fmt.Printf("Mean: %.2f, Median: %.2f\n", stats.Mean, stats.Median)
func calculateStatistics(values []float64) *Statistics {
	if len(values) == 0 {
		return &Statistics{}
	}

	sort.Float64s(values)

	stats := &Statistics{
		Min: values[0],
		Max: values[len(values)-1],
	}

	// Calculate mean.
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	stats.Mean = sum / float64(len(values))

	// Calculate median.
	mid := len(values) / 2
	if len(values)%2 == 0 {
		stats.Median = (values[mid-1] + values[mid]) / 2
	} else {
		stats.Median = values[mid]
	}

	// Calculate standard deviation.
	variance := 0.0
	for _, v := range values {
		variance += math.Pow(v-stats.Mean, 2)
	}
	stats.StdDev = math.Sqrt(variance / float64(len(values)))

	// Calculate percentiles.
	p95Index := int(float64(len(values)) * 0.95)
	p99Index := int(float64(len(values)) * 0.99)

	if p95Index < len(values) {
		stats.P95 = values[p95Index]
	} else {
		stats.P95 = values[len(values)-1]
	}

	if p99Index < len(values) {
		stats.P99 = values[p99Index]
	} else {
		stats.P99 = values[len(values)-1]
	}

	return stats
}

// calculateTrends identifies performance trends using linear regression.
//
// Arguments:
// - history: Chronologically sorted test results.
//
// Returns:
// - Trend analysis showing direction and magnitude of changes.
//
// @example
// trends := r.calculateTrends(history)
//
//	if trends.DurationTrend > 5.0 {
//	    fmt.Println("Significant performance degradation detected!")
//	}
func (r *RegressionAnalyzer) calculateTrends(history []*TestResult) *TrendAnalysis {
	if len(history) < 2 {
		return &TrendAnalysis{}
	}

	// Use recent window for trend calculation.
	windowSize := r.toleranceConfig.WindowSize
	if len(history) < windowSize {
		windowSize = len(history)
	}

	recent := history[len(history)-windowSize:]

	trends := &TrendAnalysis{
		TrendWindow: windowSize,
	}

	// Calculate duration trend.
	durations := make([]float64, len(recent))
	for i, result := range recent {
		durations[i] = float64(result.Duration.Nanoseconds())
	}
	trends.DurationTrend = calculateLinearTrend(durations)

	// Calculate memory trend.
	memories := make([]float64, len(recent))
	for i, result := range recent {
		memories[i] = float64(result.MemoryUsage)
	}
	trends.MemoryTrend = calculateLinearTrend(memories)

	// Calculate contour trend.
	contours := make([]float64, len(recent))
	for i, result := range recent {
		contours[i] = float64(result.Contours)
	}
	trends.ContourTrend = calculateLinearTrend(contours)

	return trends
}

// calculateLinearTrend computes the slope of a linear regression line.
//
// Arguments:
// - values: Time-series data points.
//
// Returns:
// - Slope indicating trend direction and magnitude (percentage change per step).
//
// @example
// trend := calculateLinearTrend([]float64{100, 102, 104, 106})
// fmt.Printf("Trend: %.2f%% per execution\n", trend)
func calculateLinearTrend(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	n := float64(len(values))
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0

	for i, y := range values {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Calculate slope.
	denominator := n*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0
	}

	slope := (n*sumXY - sumX*sumY) / denominator

	// Convert to percentage change relative to mean.
	mean := sumY / n
	if mean == 0 {
		return 0
	}

	return (slope / mean) * 100
}

// detectRegressions analyzes metrics and trends to identify regressions.
//
// Arguments:
// - report: Analysis report to populate with regression findings.
// - history: Historical test results for comparison.
//
// Returns:
// - None (modifies report in place).
//
// @example
// r.detectRegressions(report, history)
//
//	if report.HasRegression {
//	    panic("Build failed due to regression!")
//	}
func (r *RegressionAnalyzer) detectRegressions(report *AnalysisReport, history []*TestResult) {
	if len(history) < r.toleranceConfig.MinSampleSize {
		report.Summary = fmt.Sprintf("Insufficient data (need %d samples, have %d)",
			r.toleranceConfig.MinSampleSize, len(history))
		return
	}

	latest := history[len(history)-1]

	// Compare latest against baseline window.
	windowSize := r.toleranceConfig.WindowSize
	if len(history)-1 < windowSize {
		windowSize = len(history) - 1
	}

	baseline := history[len(history)-windowSize-1 : len(history)-1]

	// Calculate baseline statistics.
	baselineDurations := make([]float64, len(baseline))
	for i, result := range baseline {
		baselineDurations[i] = float64(result.Duration.Nanoseconds())
	}
	baselineStats := calculateStatistics(baselineDurations)

	// Check for duration regression.
	latestDuration := float64(latest.Duration.Nanoseconds())
	durationIncrease := ((latestDuration - baselineStats.Mean) / baselineStats.Mean) * 100

	regressions := []string{}
	improvements := []string{}

	if durationIncrease > r.toleranceConfig.DurationPercent {
		report.HasRegression = true
		regressions = append(regressions,
			fmt.Sprintf("Duration increased by %.1f%% (threshold: %.1f%%)",
				durationIncrease, r.toleranceConfig.DurationPercent))
	} else if durationIncrease < -r.toleranceConfig.DurationPercent {
		report.HasImprovement = true
		improvements = append(improvements,
			fmt.Sprintf("Duration improved by %.1f%%", -durationIncrease))
	}

	// Check for consistent negative trends.
	if report.Trends.DurationTrend > r.toleranceConfig.DurationPercent/float64(windowSize) {
		report.HasRegression = true
		regressions = append(regressions,
			fmt.Sprintf("Consistent degradation trend: %.1f%% per execution",
				report.Trends.DurationTrend))
	}

	// Build summary.
	if report.HasRegression {
		report.Summary = "REGRESSION: " + strings.Join(regressions, "; ")
	} else if report.HasImprovement {
		report.Summary = "IMPROVEMENT: " + strings.Join(improvements, "; ")
	} else {
		report.Summary = fmt.Sprintf("STABLE: Performance within %.1f%% tolerance",
			r.toleranceConfig.DurationPercent)
	}
}

// generateRecommendations provides actionable suggestions based on analysis.
//
// Arguments:
// - report: Analysis report containing metrics and trends.
//
// Returns:
// - Slice of actionable recommendations for addressing issues.
//
// @example
// recommendations := r.generateRecommendations(report)
//
//	for _, rec := range recommendations {
//	    fmt.Printf("• %s\n", rec)
//	}
func (r *RegressionAnalyzer) generateRecommendations(report *AnalysisReport) []string {
	recommendations := []string{}

	if report.HasRegression {
		recommendations = append(recommendations,
			"Review recent code changes for performance impacts",
			"Run profiling to identify bottlenecks",
			"Consider reverting recent changes if regression is severe")
	}

	if report.Metrics != nil && report.Metrics.DurationStats != nil {
		// High variance suggests instability.
		cv := report.Metrics.DurationStats.StdDev / report.Metrics.DurationStats.Mean
		if cv > 0.3 {
			recommendations = append(recommendations,
				fmt.Sprintf("High variance detected (CV=%.2f), consider stabilizing test environment", cv))
		}

		// Check success rate.
		if report.Metrics.SuccessRate < 0.95 {
			recommendations = append(recommendations,
				fmt.Sprintf("Success rate is %.1f%%, investigate failing cases",
					report.Metrics.SuccessRate*100))
		}
	}

	if report.Trends != nil {
		if math.Abs(report.Trends.ContourTrend) > r.toleranceConfig.ContourDeltaPercent {
			recommendations = append(recommendations,
				"Significant change in contour detection, verify algorithm correctness")
		}
	}

	if len(recommendations) == 0 {
		recommendations = append(recommendations,
			"Performance is stable, continue monitoring")
	}

	return recommendations
}

// GenerateHTMLReport creates an HTML visualization of the analysis.
//
// Arguments:
// - report: Analysis report to visualize.
//
// Returns:
// - HTML string for browser rendering, error if generation fails.
//
// @example
// html, err := analyzer.GenerateHTMLReport(report)
// os.WriteFile("report.html", []byte(html), 0644)
func (r *RegressionAnalyzer) GenerateHTMLReport(report *AnalysisReport) (string, error) {
	html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Regression Analysis: %s</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #333; color: white; padding: 10px; }
        .regression { background: #ffcccc; padding: 10px; margin: 10px 0; }
        .improvement { background: #ccffcc; padding: 10px; margin: 10px 0; }
        .stable { background: #ccccff; padding: 10px; margin: 10px 0; }
        .metrics { display: flex; justify-content: space-around; }
        .metric-box { border: 1px solid #ddd; padding: 10px; margin: 5px; }
        .recommendations { background: #ffffcc; padding: 10px; margin: 10px 0; }
        table { border-collapse: collapse; width: 100%%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>%s - Analysis Report</h1>
        <p>Generated: %s</p>
    </div>
    `, report.TestName, report.TestName, report.Timestamp.Format(time.RFC3339))

	// Add status summary.
	statusClass := "stable"
	if report.HasRegression {
		statusClass = "regression"
	} else if report.HasImprovement {
		statusClass = "improvement"
	}

	html += fmt.Sprintf(`
    <div class="%s">
        <h2>Status: %s</h2>
        <p>%s</p>
    </div>
    `, statusClass, strings.ToUpper(statusClass), report.Summary)

	// Add metrics.
	if report.Metrics != nil {
		html += `
    <h2>Performance Metrics</h2>
    <div class="metrics">
        <div class="metric-box">
            <h3>Duration</h3>
            <table>
                <tr><th>Mean</th><td>%.2f ms</td></tr>
                <tr><th>Median</th><td>%.2f ms</td></tr>
                <tr><th>Std Dev</th><td>%.2f ms</td></tr>
                <tr><th>P95</th><td>%.2f ms</td></tr>
                <tr><th>P99</th><td>%.2f ms</td></tr>
            </table>
        </div>
    </div>
    `
		html = fmt.Sprintf(html,
			report.Metrics.DurationStats.Mean/1e6,
			report.Metrics.DurationStats.Median/1e6,
			report.Metrics.DurationStats.StdDev/1e6,
			report.Metrics.DurationStats.P95/1e6,
			report.Metrics.DurationStats.P99/1e6)
	}

	// Add trends.
	if report.Trends != nil {
		html += fmt.Sprintf(`
    <h2>Performance Trends</h2>
    <table>
        <tr><th>Metric</th><th>Trend</th><th>Interpretation</th></tr>
        <tr>
            <td>Duration</td>
            <td>%.2f%%</td>
            <td>%s</td>
        </tr>
    </table>
    `, report.Trends.DurationTrend,
			interpretTrend(report.Trends.DurationTrend, "slower", "faster"))
	}

	// Add recommendations.
	if len(report.Recommendations) > 0 {
		html += `
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>`
		for _, rec := range report.Recommendations {
			html += fmt.Sprintf("\n            <li>%s</li>", rec)
		}
		html += `
        </ul>
    </div>`
	}

	html += `
</body>
</html>`

	return html, nil
}

// interpretTrend provides human-readable interpretation of trend values.
//
// Arguments:
// - trend: Trend value as percentage.
// - positiveLabel: Label for positive trend.
// - negativeLabel: Label for negative trend.
//
// Returns:
// - Human-readable trend interpretation.
//
// @example
// interpretation := interpretTrend(5.2, "degrading", "improving")
// // Returns: "Getting degrading by 5.2% per execution"
func interpretTrend(trend float64, positiveLabel, negativeLabel string) string {
	if math.Abs(trend) < 0.1 {
		return "Stable"
	}
	if trend > 0 {
		return fmt.Sprintf("Getting %s by %.1f%% per execution", positiveLabel, trend)
	}
	return fmt.Sprintf("Getting %s by %.1f%% per execution", negativeLabel, -trend)
}

// ExportResults exports analysis results to various formats.
//
// Arguments:
// - report: Analysis report to export.
// - format: Export format ("json", "csv", "markdown").
// - filepath: Destination file path.
//
// Returns:
// - Error if export fails, nil otherwise.
//
// @example
// err := analyzer.ExportResults(report, "markdown", "./reports/analysis.md")
//
//	if err != nil {
//	    log.Fatal(err)
//	}
func (r *RegressionAnalyzer) ExportResults(report *AnalysisReport, format, filepath string) error {
	var data []byte
	var err error

	switch strings.ToLower(format) {
	case "json":
		data, err = json.MarshalIndent(report, "", "  ")
	case "csv":
		data = []byte(r.generateCSV(report))
	case "markdown":
		data = []byte(r.generateMarkdown(report))
	case "html":
		html, htmlErr := r.GenerateHTMLReport(report)
		data = []byte(html)
		err = htmlErr
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}

	if err != nil {
		return fmt.Errorf("failed to generate %s: %w", format, err)
	}

	return os.WriteFile(filepath, data, 0o644)
}

// generateCSV creates CSV representation of the analysis report.
//
// Arguments:
// - report: Analysis report to convert to CSV.
//
// Returns:
// - CSV formatted string.
//
// @example
// csv := r.generateCSV(report)
// fmt.Print(csv)
func (r *RegressionAnalyzer) generateCSV(report *AnalysisReport) string {
	var csv strings.Builder

	csv.WriteString("Metric,Value\n")
	csv.WriteString(fmt.Sprintf("Test Name,%s\n", report.TestName))
	csv.WriteString(fmt.Sprintf("Timestamp,%s\n", report.Timestamp.Format(time.RFC3339)))
	csv.WriteString(fmt.Sprintf("Has Regression,%v\n", report.HasRegression))
	csv.WriteString(fmt.Sprintf("Has Improvement,%v\n", report.HasImprovement))

	if report.Metrics != nil {
		csv.WriteString(fmt.Sprintf("Success Rate,%.2f\n", report.Metrics.SuccessRate))
		csv.WriteString(fmt.Sprintf("Total Executions,%d\n", report.Metrics.TotalExecutions))

		if report.Metrics.DurationStats != nil {
			csv.WriteString(fmt.Sprintf("Duration Mean (ms),%.2f\n",
				report.Metrics.DurationStats.Mean/1e6))
			csv.WriteString(fmt.Sprintf("Duration StdDev (ms),%.2f\n",
				report.Metrics.DurationStats.StdDev/1e6))
		}
	}

	return csv.String()
}

// generateMarkdown creates Markdown representation of the analysis report.
//
// Arguments:
// - report: Analysis report to convert to Markdown.
//
// Returns:
// - Markdown formatted string.
//
// @example
// markdown := r.generateMarkdown(report)
// os.WriteFile("report.md", []byte(markdown), 0644)
func (r *RegressionAnalyzer) generateMarkdown(report *AnalysisReport) string {
	var md strings.Builder

	md.WriteString(fmt.Sprintf("# Regression Analysis: %s\n\n", report.TestName))
	md.WriteString(fmt.Sprintf("**Generated:** %s\n\n", report.Timestamp.Format(time.RFC3339)))

	// Status.
	status := "✅ STABLE"
	if report.HasRegression {
		status = "❌ REGRESSION"
	} else if report.HasImprovement {
		status = "🎉 IMPROVEMENT"
	}
	md.WriteString(fmt.Sprintf("## Status: %s\n\n", status))
	md.WriteString(fmt.Sprintf("%s\n\n", report.Summary))

	// Metrics.
	if report.Metrics != nil {
		md.WriteString("## Performance Metrics\n\n")
		md.WriteString("| Metric | Value |\n")
		md.WriteString("|--------|-------|\n")
		md.WriteString(fmt.Sprintf("| Success Rate | %.1f%% |\n",
			report.Metrics.SuccessRate*100))
		md.WriteString(fmt.Sprintf("| Total Executions | %d |\n",
			report.Metrics.TotalExecutions))

		if report.Metrics.DurationStats != nil {
			md.WriteString(fmt.Sprintf("| Duration Mean | %.2f ms |\n",
				report.Metrics.DurationStats.Mean/1e6))
			md.WriteString(fmt.Sprintf("| Duration P95 | %.2f ms |\n",
				report.Metrics.DurationStats.P95/1e6))
			md.WriteString(fmt.Sprintf("| Duration P99 | %.2f ms |\n",
				report.Metrics.DurationStats.P99/1e6))
		}
		md.WriteString("\n")
	}

	// Recommendations.
	if len(report.Recommendations) > 0 {
		md.WriteString("## Recommendations\n\n")
		for _, rec := range report.Recommendations {
			md.WriteString(fmt.Sprintf("- %s\n", rec))
		}
	}

	return md.String()
}

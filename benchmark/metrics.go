// Package benchmark - Functionality for running benchmarks.
package benchmark

import "time"

// PerformanceMetrics captures detailed performance data
type PerformanceMetrics struct {
	Scenario            Scenario      `json:"scenario"`
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

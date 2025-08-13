# Images Kernels Stream Processing


### [`TestVideoStreamProcessing`](../images/kernels/integration_test.go)

This test coverage shows the frame budget, tolerance settings, and exact timing that causes frames to be dropped.


```ts
../go-ml/images/kernels 🌱 main [✘!?] ➜ go test -v -run TestVideoStreamProcessing
=== RUN   TestVideoStreamProcessing
=== RUN   TestVideoStreamProcessing/YOLO_HD_30fps
Frame 1 DROPPED: processing=39.551ms > budget=33ms+tolerance=3.3ms (exceeded by 3.251ms)
Frame 16 DROPPED: processing=77.492542ms > budget=33ms+tolerance=3.3ms (exceeded by 41.192542ms)
Frame 17 DROPPED: processing=36.612833ms > budget=33ms+tolerance=3.3ms (exceeded by 312.833µs)
    integration_test.go:702: YOLO_HD_30fps Results:
          Frames: 59 processed, 3 dropped (5.08%)
          Throughput: 29.5 FPS
          Latency: avg=29.321252ms, p95=32.851417ms, p99=74.585042ms
          Memory: 1349.69 MB allocated, 119 GC pauses
          Drop Details: Frame processing exceeded 33ms budget + 10% tolerance
=== RUN   TestVideoStreamProcessing/DFINE_FHD_30fps
Frame 21 DROPPED: processing=56.227166ms > budget=33ms+tolerance=3.3ms (exceeded by 19.927166ms)
    integration_test.go:702: DFINE_FHD_30fps Results:
          Frames: 60 processed, 1 dropped (1.67%)
          Throughput: 30.0 FPS
          Latency: avg=26.314083ms, p95=26.884959ms, p99=49.598666ms
          Memory: 1224.87 MB allocated, 90 GC pauses
          Drop Details: Frame processing exceeded 33ms budget + 10% tolerance
=== RUN   TestVideoStreamProcessing/FasterRCNN_HD_15fps
    integration_test.go:702: FasterRCNN_HD_15fps Results:
          Frames: 30 processed, 0 dropped (0.00%)
          Throughput: 15.0 FPS
          Latency: avg=29.720993ms, p95=30.7475ms, p99=32.524875ms
          Memory: 688.04 MB allocated, 59 GC pauses
=== RUN   TestVideoStreamProcessing/RTDETR_HD_30fps
    integration_test.go:702: RTDETR_HD_30fps Results:
          Frames: 61 processed, 0 dropped (0.00%)
          Throughput: 30.5 FPS
          Latency: avg=28.411519ms, p95=29.212584ms, p99=30.225833ms
          Memory: 1392.08 MB allocated, 122 GC pauses
--- PASS: TestVideoStreamProcessing (8.06s)
    --- PASS: TestVideoStreamProcessing/YOLO_HD_30fps (2.01s)
    --- PASS: TestVideoStreamProcessing/DFINE_FHD_30fps (2.01s)
    --- PASS: TestVideoStreamProcessing/FasterRCNN_HD_15fps (2.01s)
    --- PASS: TestVideoStreamProcessing/RTDETR_HD_30fps (2.03s)
PASS
ok      github.com/nvr-ai/go-ml/images/kernels  8.246s
```

## Analysis

1. **YOLO_HD_30fps**: 3 frames dropped (5.08%)
   - Frame 1: 39.551ms processing > **33ms+3.3ms** budget (exceeded by **3.251ms**)
   - Frame 16: 77.492ms processing > **33ms+3.3ms** budget (exceeded by **41.192ms**)
   - Frame 17: 36.613ms processing > **33ms+3.3ms** budget (exceeded by **312µs**)

2. **DFINE_FHD_30fps**: 1 frame dropped (1.67%)
   - Frame 21: 56.227ms processing > **33ms+3.3ms** budget (exceeded by **19.927ms**)

3. **FasterRCNN_HD_15fps**: 0 frames dropped (0.00%) 
   - All frames processed within **66.7ms** budget (15fps target)

4. **RTDETR_HD_30fps**: 0 frames dropped (0.00%) 
   - All frames processed within **33ms+3.3ms** budget (30fps target)

**Performance Analysis:**
- Frame drops occur when processing exceeds the **33ms budget + 10% tolerance (3.3ms)** 
- YOLO shows highest variability with severe spikes (**77.5ms peak**)
- DFINE has moderate spikes but better consistency
- FasterRCNN benefits from relaxed 15fps timing requirements
- RTDETR demonstrates most consistent performance within budget (**30fps target**)  
- Memory pressure correlates with frame drops: YOLO (1349MB, 119 GC) vs RTDETR (1392MB, 122 GC)

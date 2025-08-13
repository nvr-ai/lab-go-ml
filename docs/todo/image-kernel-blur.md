# Image Kernel Blur Tasks

| Date       | Description            |
| ---------- | ---------------------- |
| 2025-08-12 | Initial draft proofed. |

## Configuration

### Memory Pool Configuration

```go
// Enable memory pooling for all video processing
pool := &kernels.Pool{}
opts := kernels.Options{
    Edge:     kernels.EdgeClamp,
    Radius:   1,       // Model-specific (see below)
    Pool:     pool,    // CRITICAL: Always enable
    Parallel: true,    // Multi-core utilization
}
```

### Model-Specific Blur Settings

#### Kernel Options

```go
// Enable memory pooling for all video processing
pool := &kernels.Pool{}
opts := kernels.Options{
    Edge:     kernels.EdgeClamp,
    Radius:   1,       // Model-specific (see below)
    Pool:     pool,    // CRITICAL: Always enable
    Parallel: true,    // Multi-core utilization
}
```

#### Blur Configurations

```go
var blurConfig = map[string]int{
    "d-fine":      0,  // NEVER blur - corrupts deformable attention
    "yolov4":      1,  // Minimal - preserves small object detection
    "yolov8":      1,  // Minimal - modern YOLO architectures
    "faster-rcnn": 2,  // Moderate - RPN tolerates some blur
    "rt-detr":     0,  // Transformers are precision-sensitive
}
```

## Optimizations

### SIMD Vectorization

- [ ] Opportunity: Process 4 pixels simultaneously using AVX2.
- [ ] Expected speedup: Additional 4-8× performance improvement.
- [ ] Implementation: Use unsafe.Pointer + SIMD intrinsics.

### GPU Acceleration

- [ ] For ultra-high throughput scenarios (>1000 concurrent streams).
- [ ] Expected performance: <0.1ms for 1920×1080 on RTX 4090.
- [ ] ROI: Justified only for massive scale deployments.

### Adaptive Blur Selection

- [ ] Dynamic blur based on motion detection and object density.
- [ ] High motion: radius=0 (preserve temporal consistency).
- [ ] Low motion: radius=1-2 (noise reduction acceptable).

### Alternative Algorithms

- [ ] Gaussian approximation via multiple box blurs.
- [ ] Quality: Superior to single box blur.
- [ ] Performance: Similar to current GPT5 implementation.
- [ ] Formula: `σ ≈ √((n(w²-1)/12))` where `n`=passes, `w`=box width.

### Hardware-Specific Optimization

Batched multi-frame processing if GPU is available.

### Content-Aware Processing

- [ ] Skip blur in high-detail regions.
- [ ] Apply only where noise reduction provides benefit.
- [ ] Integration with motion vectors for temporal coherence.

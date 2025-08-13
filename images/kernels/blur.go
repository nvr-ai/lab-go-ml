package kernels

import (
	"image"
	"sync"
)

// EdgeMode defines how sampling behaves outside the image bounds.
// - Clamp: repeats edge pixels (fast, common, can darken edges slightly).
// - Mirror: reflects coordinates (better edge energy preservation).
// - Wrap: tiles the image (for periodic patterns).
type EdgeMode int

const (
	EdgeClamp EdgeMode = iota
	EdgeMirror
	EdgeWrap
)

// Options configures the blur call. Keeping this extensible reduces churn later.
type Options struct {
	Radius   int      // Blur radius (window size = 2*Radius + 1). Must be >= 0.
	Edge     EdgeMode // Edge sampling mode.
	Pool     *Pool    // Optional buffer pool for intermediate/dst reuse.
	Parallel bool     // Enable row/column parallelism (good for 1080p+).
}

// Pool lets callers reuse large buffers to reduce GC pressure at 30–60 FPS video rates.
type Pool struct {
	rgba sync.Pool // *image.RGBA
}

func (p *Pool) GetRGBA(bounds image.Rectangle) *image.RGBA {
	if p == nil {
		return image.NewRGBA(bounds)
	}
	if v := p.rgba.Get(); v != nil {
		img := v.(*image.RGBA)
		if img.Rect == bounds {
			return img
		}
	}
	return image.NewRGBA(bounds)
}
func (p *Pool) PutRGBA(img *image.RGBA) {
	if p == nil || img == nil {
		return
	}
	// Clear is optional; we skip for speed. The next writer fully overwrites.
	p.rgba.Put(img)
}

// BoxBlur applies a fast, separable box blur to an image.
// This version is optimized for real-time pipelines and correctness:
// - Converts to premultiplied RGBA once and operates on raw bytes.
// - Uses a sliding window per row/col to achieve O(1) updates per pixel.
// - Avoids float math and per-sample function calls in hot loops.
// - Correctly accounts for non-zero image bounds.
//
// Performance: O(W*H) per pass, independent of Radius (excluding small edge regions).
// Quality: Lower than Gaussian; for better quality at similar speed, consider
//
//	running three box blurs with radii chosen to approximate Gaussian.
//
// Implications for object detection pipelines:
// - Pre-detector blur can reduce small-object recall; use minimal Radius or off.
// - For privacy blur (post-detection), consider blurring once and compositing ROIs.
//
// Returns a new *image.RGBA. If Options.Pool is provided, buffers may be reused.
func BoxBlur(src image.Image, opt Options) *image.RGBA {
	r := opt.Radius
	if r <= 0 {
		// Return a copy in premultiplied RGBA to make downstream usage predictable.
		b := src.Bounds()
		dst := image.NewRGBA(b)
		drawImage(dst, src) // small helper; see below
		return dst
	}

	// 1) Normalize source to premultiplied RGBA to avoid alpha correctness issues
	// and to allow raw byte access without color model overhead.
	rgbaSrc := toRGBA(src) // converts or casts; see below
	b := rgbaSrc.Rect

	// 2) Allocate intermediate and destination buffers, optionally from Pool.
	tmp := opt.Pool.GetRGBA(b)
	dst := opt.Pool.GetRGBA(b)

	// 3) Horizontal pass (row-wise), sliding window over X for each Y.
	boxBlurHorizRGBA(rgbaSrc, tmp, r, opt.Edge, opt.Parallel)

	// 4) Vertical pass (column-wise), sliding window over Y for each X.
	boxBlurVertRGBA(tmp, dst, r, opt.Edge, opt.Parallel)

	// 5) Return dst; release tmp back to pool if used.
	opt.Pool.PutRGBA(tmp)
	return dst
}

// toRGBA returns a *image.RGBA view/copy of src.
// If src already is *image.RGBA, it returns it directly.
// Otherwise, it draws src into a new RGBA, ensuring premultiplied storage.
func toRGBA(src image.Image) *image.RGBA {
	if r, ok := src.(*image.RGBA); ok {
		return r
	}
	b := src.Bounds()
	dst := image.NewRGBA(b)
	drawImage(dst, src)
	return dst
}

// drawImage copies src into dst. Using draw.Draw would be fine; we inline a minimal path.
// This preserves premultiplication semantics via Go's image/draw rules.
func drawImage(dst *image.RGBA, src image.Image) {
	// We deliberately use standard library draw to handle all color models correctly.
	// If you want zero dependency, reimplement a fast copy for RGBA/NRGBA cases.
	// For brevity and correctness, we delegate:
	// draw.Draw(dst, dst.Rect, src, src.Bounds().Min, draw.Src)
	// But we avoid importing draw in this snippet; uncomment if you use draw.
	for y := dst.Rect.Min.Y; y < dst.Rect.Max.Y; y++ {
		for x := dst.Rect.Min.X; x < dst.Rect.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
}

// BlurRegions blurs only the given regions by compositing from a blurred copy.
// This costs one full-frame blur plus a few rectangle copies.
// Regions are clipped to bounds; overlapping regions are handled naturally.
func BlurRegions(src image.Image, regions []image.Rectangle, opt Options) *image.RGBA {
	blurred := BoxBlur(src, opt) // one full blur
	// Compose onto a copy of the original
	dst := toRGBA(src).SubImage(toRGBA(src).Rect).(*image.RGBA) // make sure to copy if needed
	out := opt.Pool.GetRGBA(dst.Rect)
	copy(out.Pix, dst.Pix) // shallow copy bytes

	b := dst.Rect
	for _, r := range regions {
		r = r.Intersect(b)
		if r.Empty() {
			continue
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			srcOff := (y-blurred.Rect.Min.Y)*blurred.Stride + (r.Min.X-blurred.Rect.Min.X)*4
			dstOff := (y-out.Rect.Min.Y)*out.Stride + (r.Min.X-out.Rect.Min.X)*4
			n := (r.Dx()) * 4
			copy(out.Pix[dstOff:dstOff+n], blurred.Pix[srcOff:srcOff+n])
		}
	}
	opt.Pool.PutRGBA(blurred)
	return out
}

// boxBlurHorizRGBA applies horizontal blur into dst using a sliding window.
// Reads from src.Pix and writes to dst.Pix. Both must share the same bounds.
// The sliding window means we:
//   - Compute an initial sum for x in [-r .. +r], respecting edges.
//   - For each step to the right, we subtract the pixel leaving on the left
//     and add the pixel entering on the right. This keeps O(1) cost per pixel.
func boxBlurHorizRGBA(src, dst *image.RGBA, r int, edge EdgeMode, parallel bool) {
	b := src.Rect
	w := b.Dx()
	h := b.Dy()
	if w == 0 || h == 0 {
		return
	}

	window := 2*r + 1
	rowTask := func(y int) {
		yAbs := b.Min.Y + y
		// Compute row start in Pix for both src and dst once to avoid PixOffset per pixel.
		srcRowStart := (yAbs - src.Rect.Min.Y) * src.Stride
		dstRowStart := (yAbs - dst.Rect.Min.Y) * dst.Stride

		// Accumulators for premultiplied RGBA channels.
		var sumR, sumG, sumB, sumA uint32

		// Helper to load a pixel at (xRel) in this row with edge handling.
		// xRel is relative [0..w-1]; we map indices outside via edge mode.
		load := func(xRel int) (r, g, b, a uint32) {
			xMap := mapCoord(xRel, w, edge)
			off := srcRowStart + xMap*4
			p := src.Pix[off : off+4 : off+4]
			return uint32(p[0]), uint32(p[1]), uint32(p[2]), uint32(p[3])
		}

		// 1) Initialize window sum for x=0: sum over [-r..+r].
		sumR, sumG, sumB, sumA = 0, 0, 0, 0
		for dx := -r; dx <= r; dx++ {
			r8, g8, b8, a8 := load(dx)
			sumR += r8
			sumG += g8
			sumB += b8
			sumA += a8
		}

		// 2) Slide across x from 0..w-1.
		// Integer division by constant window is cheap; Go will optimize it.
		for x := 0; x < w; x++ {
			dstOff := dstRowStart + x*4
			dst.Pix[dstOff+0] = uint8((sumR + uint32(window/2)) / uint32(window))
			dst.Pix[dstOff+1] = uint8((sumG + uint32(window/2)) / uint32(window))
			dst.Pix[dstOff+2] = uint8((sumB + uint32(window/2)) / uint32(window))
			dst.Pix[dstOff+3] = uint8((sumA + uint32(window/2)) / uint32(window))

			// Next window: remove left, add right.
			// Left sample exits at x - r
			lr, lg, lb, la := load(x - r)
			// Right sample enters at x + r + 1
			rr, rg, rb, ra := load(x + r + 1)
			// Update sum: new = old - left + right
			sumR += rr - lr
			sumG += rg - lg
			sumB += rb - lb
			sumA += ra - la
		}
	}

	if !parallel || h < 4 {
		for y := 0; y < h; y++ {
			rowTask(y)
		}
		return
	}

	// Parallelize by splitting rows into chunks.
	// Choose chunk size to avoid too many goroutines and preserve cache locality.
	chunk := chooseChunk(h)
	var wg sync.WaitGroup
	for start := 0; start < h; start += chunk {
		end := start + chunk
		if end > h {
			end = h
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for y := s; y < e; y++ {
				rowTask(y)
			}
		}(start, end)
	}
	wg.Wait()
}

// boxBlurVertRGBA mirrors the horizontal pass but along columns.
// We reuse the same sliding-window logic, but now Y changes and we stride by src.Stride per step.
func boxBlurVertRGBA(src, dst *image.RGBA, r int, edge EdgeMode, parallel bool) {
	b := src.Rect
	w := b.Dx()
	h := b.Dy()
	if w == 0 || h == 0 {
		return
	}

	window := 2*r + 1
	colTask := func(x int) {
		xAbs := b.Min.X + x
		// Helper to load a pixel at (yRel) in this column with edge handling.
		load := func(yRel int) (r, g, b, a uint32) {
			yMap := mapCoord(yRel, h, edge)
			srcRowStart := yMap * src.Stride
			off := srcRowStart + x*4
			p := src.Pix[off : off+4 : off+4]
			return uint32(p[0]), uint32(p[1]), uint32(p[2]), uint32(p[3])
		}

		// Initialize window sum for y=0.
		var sumR, sumG, sumB, sumA uint32
		for dy := -r; dy <= r; dy++ {
			r8, g8, b8, a8 := load(dy)
			sumR += r8
			sumG += g8
			sumB += b8
			sumA += a8
		}

		for y := 0; y < h; y++ {
			dstRowStart := y * dst.Stride // b aligns dst.Rect; safe since dst shares b
			dstOff := dstRowStart + x*4
			dst.Pix[dstOff+0] = uint8((sumR + uint32(window/2)) / uint32(window))
			dst.Pix[dstOff+1] = uint8((sumG + uint32(window/2)) / uint32(window))
			dst.Pix[dstOff+2] = uint8((sumB + uint32(window/2)) / uint32(window))
			dst.Pix[dstOff+3] = uint8((sumA + uint32(window/2)) / uint32(window))

			// Slide: remove above, add below
			lr, lg, lb, la := load(y - r)
			rr, rg, rb, ra := load(y + r + 1)
			sumR += rr - lr
			sumG += rg - lg
			sumB += rb - lb
			sumA += ra - la
		}
		_ = xAbs // documented to show absolute coordinates; not needed in slices.
	}

	if !parallel || w < 4 {
		for x := 0; x < w; x++ {
			colTask(x)
		}
		return
	}
	chunk := chooseChunk(w)
	var wg sync.WaitGroup
	for start := 0; start < w; start += chunk {
		end := start + chunk
		if end > w {
			end = w
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for x := s; x < e; x++ {
				colTask(x)
			}
		}(start, end)
	}
	wg.Wait()
}

// mapCoord maps an index i to [0, n) according to edge mode.
// Designed to be inlineable and branch-light.
// For Clamp: clamp to [0, n-1].
// For Mirror: reflect indices ... -2,-1,0,1,2, ... -> 1,0,0,1,2, ... (no duplication at edges).
// For Wrap: modulo wrap to [0, n).
func mapCoord(i, n int, mode EdgeMode) int {
	switch mode {
	case EdgeClamp:
		if i < 0 {
			return 0
		}
		if i >= n {
			return n - 1
		}
		return i
	case EdgeMirror:
		// Mirror within [0, n-1] with no double-counting of edge.
		if n == 1 {
			return 0
		}
		for i < 0 || i >= n {
			if i < 0 {
				i = -i - 1
			} else if i >= n {
				i = 2*n - i - 1
			}
		}
		return i
	case EdgeWrap:
		if n == 0 {
			return 0
		}
		i %= n
		if i < 0 {
			i += n
		}
		return i
	default:
		// Fallback to clamp.
		if i < 0 {
			return 0
		}
		if i >= n {
			return n - 1
		}
		return i
	}
}

// chooseChunk picks a work chunk size that balances overhead and cache locality.
// You can tune this per CPU; here we use a simple heuristic.
func chooseChunk(n int) int {
	switch {
	case n >= 2048:
		return 128
	case n >= 512:
		return 64
	default:
		return 32
	}
}

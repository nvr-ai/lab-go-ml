package inference

import (
	"fmt"
	"image"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/nfnt/resize"
)

// PrepareInput prepares the input for the ONNX model before inference is
// called typically before the predict() function calls.
//
// Arguments:
//   - img: The image to prepare.
//   - dst: The destination tensor to populate.
//
// Returns:
//   - error: An error if the input preparation fails.
func PrepareInput(img image.Image, dst *ort.Tensor[float32]) error {
	data := dst.GetData()
	channelSize := 640 * 640
	if len(data) < (channelSize * 3) {
		return fmt.Errorf("Destination tensor only holds %d floats, needs "+
			"%d (make sure it's the right shape!)", len(data), channelSize*3)
	}
	red := data[0:channelSize]
	green := data[channelSize : channelSize*2]
	blue := data[channelSize*2 : channelSize*3]

	// Resize the image to 640x640 using Lanczos3 algorithm.
	img = resize.Resize(640, 640, img, resize.Lanczos3)

	i := 0
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			red[i] = float32(r>>8) / 255.0
			green[i] = float32(g>>8) / 255.0
			blue[i] = float32(b>>8) / 255.0
			i++
		}
	}
	return nil
}

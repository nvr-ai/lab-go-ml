package dfine

import (
	"fmt"

	"github.com/nfnt/resize"
	"github.com/nvr-ai/go-ml/images"
)

// PreProcess preprocesses the input image for the D-FINE model.
func (p *DFINE) PreProcess(input []images.Image) []float32 {
	channelSize := 640 * 640
	if len(data) < (channelSize * 3) {
		return fmt.Errorf("Destination tensor only holds %d floats, needs "+
			"%d (make sure it's the right shape!)", len(data), channelSize*3)
	}
	red := data[0:channelSize]
	green := data[channelSize : channelSize*2]
	blue := data[channelSize*2 : channelSize*3]

	// Resize the image to 640x640 using Lanczos3 algorithm.
	img = resize.Resize(640, 640, input, resize.Lanczos3)

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

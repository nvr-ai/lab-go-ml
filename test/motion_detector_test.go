package test

import (
	"fmt"
	"image"
	"testing"
	"time"

	"github.com/nvr-ai/go-ml/images"
	"github.com/stretchr/testify/require"
	"gocv.io/x/gocv"
)

func TestMotionDetector(t *testing.T) {
	sourceDir := "../../../ml/corpus/images/clip-4k.mp4"
	imageFiles, err := LoadDirectoryImageFiles(sourceDir)
	require.NoError(t, err)
	require.Greater(t, len(imageFiles), 20, "Need at least 20 frames for comprehensive motion detection test")

	testConfigs := []struct {
		name   string
		config motion.Config
	}{
		{
			name: "high_sensitivity",
			config: motion.Config{
				MinimumArea:       5000,
				MinMotionDuration: 50 * time.Millisecond,
			},
		},
		// {
		// 	name: "medium_sensitivity",
		// 	config: motion.Config{
		// 		MinimumArea:       20000,
		// 		MinMotionDuration: 200 * time.Millisecond,
		// 	},
		// },
		// {
		// 	name: "low_sensitivity",
		// 	config: motion.Config{
		// 		MinimumArea:       50000,
		// 		MinMotionDuration: 500 * time.Millisecond,
		// 	},
		// },
	}

	configResults := make(map[string]interface{})

	for _, testConfig := range testConfigs {
		t.Run(testConfig.name, func(t *testing.T) {
			detector := motion.New(testConfig.config)
			defer detector.Close()

			detector.Process(true, 0)

			segmenter := images.NewMotionSegmenter()
			defer segmenter.Close()

			segmenter.Kernel = gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
			defer segmenter.Kernel.Close()

			fmt.Printf("path=%s,frames=%d\n", sourceDir, len(imageFiles))
			// Process frame sequence
			for i, img := range imageFiles {
				img := images.MotionSegmenterInput{
					Data:     img.Data,
					ReadFlag: gocv.IMReadGrayScale,
				}
				if err := segmenter.Set(img); err != nil {
					t.Fatalf("Failed to set frame %d: %v", i, err)
				}

				contours := segmenter.SegmentMotion(frame)
			}
		})
	}
}

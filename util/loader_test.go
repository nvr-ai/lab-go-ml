package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadDirectoryImages(t *testing.T) {
	images, err := LoadDirectoryImageFiles("../../../ml/corpus/images/clip-4k.mp4")
	if err != nil {
		t.Fatalf("Failed to load images: %v", err)
	}

	assert.Equal(t, len(images), 823)

	for _, image := range images {
		assert.Greater(t, len(image.Data), 5000)
		// t.Logf("path=%s,frame=%d/%d,size=%d bytes\n", image.Path, image.Frame, images[len(images)-1].Frame, len(image.Data))
	}
}

package util

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ImageFile represents an image file.
type ImageFile struct {
	// Path is the path to the image file.
	Path string
	// Data is the raw bytes of the image file.
	Data []byte
	// Frame is the frame number of the image file.
	Frame int
}

// LoadDirectoryImageFiles reads all image files from a directory.
//
// Arguments:
// - dir: Directory path containing image files.
//
// Returns:
// - []ImageFile: Slice of ImageFile, each containing the raw bytes of an image file.
// - error: Error if loading fails.
func LoadDirectoryImageFiles(dir string) ([]ImageFile, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var images []ImageFile
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		ext := filepath.Ext(file.Name())
		switch ext {
		case ".jpg", ".jpeg", ".png", ".bmp":
			imgPath := filepath.Join(dir, file.Name())
			data, readErr := os.ReadFile(imgPath)
			if readErr != nil {
				return nil, readErr
			}
			frame, err := strconv.Atoi(strings.TrimSuffix(strings.ReplaceAll(file.Name(), "frame-", ""), ext))
			if err != nil {
				return nil, err
			}
			images = append(images, ImageFile{
				Path:  imgPath,
				Data:  data,
				Frame: frame,
			})
		}
	}

	sort.Slice(images, func(i, j int) bool {
		return images[i].Frame < images[j].Frame
	})

	return images, nil
}

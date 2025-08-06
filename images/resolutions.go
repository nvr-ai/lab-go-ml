// Package images provides type definitions and constants for common security
// surveillance camera resolutions. The definitions are designed for production-grade
// readiness, ensuring accuracy, consistency, and maintainability.
package images

import (
	"fmt"
	"math"
)

// AspectRatio represents a CCTV aspect ratio by name (e.g., "16:9").
type AspectRatio string

// Defines standard and common aspect ratios for surveillance cameras.
const (
	AspectRatio169 AspectRatio = "16:9"
	AspectRatio43  AspectRatio = "4:3"
	AspectRatio54  AspectRatio = "5:4"
	AspectRatio179 AspectRatio = "17:9" // Common in some high-end sensors
)

// ResolutionType represents a common name or standard for a CCTV resolution.
type ResolutionType string

// Defines the unique type for each supported camera resolution.
// This provides a clear, non-ambiguous identifier for each resolution standard.
const (
	ResolutionTypeNHD      ResolutionType = "nHD"
	ResolutionTypeFWVGA    ResolutionType = "FWVGA"
	ResolutionTypeQHD540   ResolutionType = "qHD 540p"
	ResolutionTypeHD720p   ResolutionType = "HD 720p"
	ResolutionTypeWXGA     ResolutionType = "WXGA"
	ResolutionTypeHDPlus   ResolutionType = "HD+"
	ResolutionType1MP54    ResolutionType = "1MP (5:4)"
	ResolutionTypeFHD1080p ResolutionType = "Full HD 1080p"
	ResolutionType2MP43    ResolutionType = "2MP (4:3)"
	ResolutionTypeQHD1440p ResolutionType = "QHD 1440p"
	ResolutionType3MP43    ResolutionType = "3MP (4:3)"
	ResolutionType4MP169   ResolutionType = "4MP (16:9)"
	ResolutionType6MP43    ResolutionType = "6MP (3:2)" // Corrected aspect ratio
	ResolutionTypeQHDPlus  ResolutionType = "QHD+"
	ResolutionType4KUHD    ResolutionType = "4K UHD"
	ResolutionType5K       ResolutionType = "5K"
	ResolutionType8KUHD    ResolutionType = "8K UHD"
	ResolutionType12MP     ResolutionType = "12MP (4:3)"
	ResolutionType16KUHD   ResolutionType = "16K UHD"
	ResolutionType32KUHD   ResolutionType = "32K UHD"
	ResolutionType64KUHD   ResolutionType = "64K UHD"
)

// ResolutionPixels describes the exact dimensions of a resolution.
type ResolutionPixels struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Resolution describes the complete set of attributes for a CCTV resolution standard.
// It includes metadata for easy identification and filtering.
type Resolution struct {
	Name         ResolutionType   `json:"name"`
	AspectRatio  AspectRatio      `json:"aspectRatio"`
	Pixels       ResolutionPixels `json:"pixels"`
	Experimental bool             `json:"experimental"` // Flags resolutions not yet in common commercial use.
}

// GetMegaPixels calculates the megapixel value based on the resolution's pixel dimensions.
// It returns the value rounded to two decimal places (e.g., 2.07 for 1080p).
// This method avoids hardcoding megapixel values, ensuring data integrity.
// O(1) complexity.
func (r Resolution) GetMegaPixels() float64 {
	if r.Pixels.Width <= 0 || r.Pixels.Height <= 0 {
		return 0.0
	}
	// Calculate total pixels and divide by 1 million for megapixels.
	mp := float64(r.Pixels.Width*r.Pixels.Height) / 1_000_000.0
	// Round to two decimal places for standardization.
	return math.Round(mp*100) / 100
}

// String returns a human-readable summary of the resolution.
// O(1) complexity.
func (r Resolution) String() string {
	return fmt.Sprintf("%s (%dx%d, %.2fMP)", r.Name, r.Pixels.Width, r.Pixels.Height, r.GetMegaPixels())
}

// resolutions is a private map that stores all defined resolution standards,
// keyed by their ResolutionType for efficient lookups.
var resolutions = map[ResolutionType]Resolution{
	ResolutionTypeNHD: {
		Name:        ResolutionTypeNHD,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 640, Height: 360},
	},
	ResolutionTypeFWVGA: {
		Name:        ResolutionTypeFWVGA,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 854, Height: 480},
	},
	ResolutionTypeQHD540: {
		Name:        ResolutionTypeQHD540,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 960, Height: 540},
	},
	ResolutionTypeHD720p: {
		Name:        ResolutionTypeHD720p,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 1280, Height: 720},
	},
	ResolutionTypeWXGA: {
		Name:        ResolutionTypeWXGA,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 1366, Height: 768},
	},
	ResolutionTypeHDPlus: {
		Name:        ResolutionTypeHDPlus,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 1600, Height: 900},
	},
	ResolutionType1MP54: {
		Name:        ResolutionType1MP54,
		AspectRatio: AspectRatio54,
		Pixels:      ResolutionPixels{Width: 1280, Height: 1024},
	},
	ResolutionTypeFHD1080p: {
		Name:        ResolutionTypeFHD1080p,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 1920, Height: 1080},
	},
	ResolutionType2MP43: {
		Name:        ResolutionType2MP43,
		AspectRatio: AspectRatio43,
		Pixels:      ResolutionPixels{Width: 1600, Height: 1200},
	},
	ResolutionTypeQHD1440p: {
		Name:        ResolutionTypeQHD1440p,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 2560, Height: 1440},
	},
	ResolutionType3MP43: {
		Name:        ResolutionType3MP43,
		AspectRatio: AspectRatio43,
		Pixels:      ResolutionPixels{Width: 2048, Height: 1536},
	},
	ResolutionType4MP169: {
		Name:        ResolutionType4MP169,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 2688, Height: 1520},
	},
	ResolutionType6MP43: {
		Name:        "6MP (3:2)", // Corrected name
		AspectRatio: "3:2",       // Corrected aspect ratio
		Pixels:      ResolutionPixels{Width: 3072, Height: 2048},
	},
	ResolutionTypeQHDPlus: {
		Name:        ResolutionTypeQHDPlus,
		AspectRatio: AspectRatio179,
		Pixels:      ResolutionPixels{Width: 3200, Height: 1800},
	},
	ResolutionType4KUHD: {
		Name:        ResolutionType4KUHD,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 3840, Height: 2160},
	},
	ResolutionType12MP: {
		Name:        ResolutionType12MP,
		AspectRatio: AspectRatio43,
		Pixels:      ResolutionPixels{Width: 4000, Height: 3000},
	},
	ResolutionType5K: {
		Name:        ResolutionType5K,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 5120, Height: 2880},
	},
	ResolutionType8KUHD: {
		Name:        ResolutionType8KUHD,
		AspectRatio: AspectRatio169,
		Pixels:      ResolutionPixels{Width: 7680, Height: 4320},
	},
	ResolutionType16KUHD: {
		Name:         ResolutionType16KUHD,
		AspectRatio:  AspectRatio169,
		Pixels:       ResolutionPixels{Width: 15360, Height: 8640},
		Experimental: true,
	},
	ResolutionType32KUHD: {
		Name:         ResolutionType32KUHD,
		AspectRatio:  AspectRatio169,
		Pixels:       ResolutionPixels{Width: 30720, Height: 17280},
		Experimental: true,
	},
	ResolutionType64KUHD: {
		Name:         ResolutionType64KUHD,
		AspectRatio:  AspectRatio169,
		Pixels:       ResolutionPixels{Width: 61440, Height: 34560},
		Experimental: true,
	},
}

// GetAllResolutions returns a slice of all defined resolution standards.
// The order is not guaranteed.
// O(N) complexity, where N is the number of resolutions.
func GetAllResolutions() []Resolution {
	all := make([]Resolution, 0, len(resolutions))
	for _, res := range resolutions {
		all = append(all, res)
	}
	return all
}

// GetSupportedResolutions returns a slice of all non-experimental resolutions.
// This is useful for populating UI elements with commercially available options.
// The order is not guaranteed.
// O(N) complexity, where N is the number of resolutions.
func GetSupportedResolutions() []Resolution {
	supported := make([]Resolution, 0, len(resolutions))
	for _, res := range resolutions {
		if !res.Experimental {
			supported = append(supported, res)
		}
	}
	return supported
}

// GetResolutionByType retrieves a specific resolution by its type.
// It returns the Resolution and true if found, otherwise an empty Resolution and false.
// O(1) complexity due to map lookup.
func GetResolutionByType(t ResolutionType) (Resolution, bool) {
	res, ok := resolutions[t]
	return res, ok
}

// GetHighestResolutionUnderDimensions retrieves the highest resolution that is under the given width and height.
// It returns the Resolution and true if found, otherwise an empty Resolution and false.
// O(N) complexity, where N is the number of resolutions.
//
// Arguments:
//   - width: The maximum possible width of the image.
//   - height: The maximum possible height of the image.
//
// Returns:
//   - Resolution: The highest resolution that is under the given width and height.
//   - bool: True if a resolution was found, otherwise false.
func GetHighestResolutionUnderDimensions(width, height int) (Resolution, bool) {
	var highest Resolution
	var found bool

	for _, res := range resolutions {
		if res.Pixels.Width <= width && res.Pixels.Height <= height {
			if !found || res.GetMegaPixels() > highest.GetMegaPixels() {
				highest = res
				found = true
			}
		}
	}
	return highest, found
}

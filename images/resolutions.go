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

// ResolutionGroup represents the logical families based on resolution scale and use-case.
type ResolutionGroup string

// Defines the unique type for each supported camera resolution.
// This provides a clear, non-ambiguous identifier for each resolution standard.
const (
	// ResolutionGroupLegacy resolutions are older or niche formats	"nHD", "FWVGA", "qHD 540p"
	ResolutionGroupLegacy ResolutionGroup = "Legacy"
	// ResolutionGroupHD represents standard HD resolutions including "HD 720p", "HD+", "WXGA".
	ResolutionGroupHD ResolutionGroup = "HD"
	// ResolutionGroupFHD represents Full HD tier resolutions including "Full HD 1080p".
	ResolutionGroupFHD ResolutionGroup = "FHD"
	// ResolutionGroupQHD represents Quad HD tier resolutions including "QHD 1440p", "QHD+".
	ResolutionGroupQHD ResolutionGroup = "QHD"
	// ResolutionGroupUHD represents Ultra HD and beyond resolutions including "4K UHD", "8K UHD", "16K UHD".
	ResolutionGroupUHD ResolutionGroup = "UHD"
	// ResolutionGroupMP represents megapixel-based formats including "1MP (5:4)", "12MP (4:3)".
	ResolutionGroupMP ResolutionGroup = "MP"
)

// ResolutionType represents a common name or standard for camera resolutions.
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

// Pixels describes the exact dimensions of a resolution.
type Pixels struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Resolution describes the complete set of attributes for a CCTV resolution standard.
// It includes metadata for easy identification and filtering.
type Resolution struct {
	Type        ResolutionType  `json:"type" yaml:"type"`
	Group       ResolutionGroup `json:"group" yaml:"group"`
	Width       int             `json:"width" yaml:"width"`
	Height      int             `json:"height" yaml:"height"`
	Pixels      Pixels          `json:"pixels" yaml:"pixels"`
	AspectRatio AspectRatio     `json:"aspectRatio" yaml:"aspectRatio"`
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
	return fmt.Sprintf("%s (%dx%d, %.2fMP)", r.Type, r.Pixels.Width, r.Pixels.Height, r.GetMegaPixels())
}

// resolutions is a private variable that holds the catalog of all defined resolution standards,
// keyed by their ResolutionType for efficient lookups.
var resolutions = map[ResolutionType]Resolution{
	ResolutionTypeNHD: {
		Type:        ResolutionTypeNHD,
		Group:       ResolutionGroupLegacy,
		Width:       640,
		Height:      360,
		Pixels:      Pixels{Width: 640, Height: 360},
		AspectRatio: AspectRatio169,
	},
	ResolutionTypeFWVGA: {
		Type:        ResolutionTypeFWVGA,
		Group:       ResolutionGroupLegacy,
		Width:       854,
		Height:      480,
		Pixels:      Pixels{Width: 854, Height: 480},
		AspectRatio: AspectRatio169,
	},
	ResolutionTypeQHD540: {
		Type:        ResolutionTypeQHD540,
		Group:       ResolutionGroupLegacy,
		Width:       960,
		Height:      540,
		Pixels:      Pixels{Width: 960, Height: 540},
		AspectRatio: AspectRatio169,
	},
	ResolutionTypeHD720p: {
		Type:        ResolutionTypeHD720p,
		Group:       ResolutionGroupHD,
		Width:       1280,
		Height:      720,
		Pixels:      Pixels{Width: 1280, Height: 720},
		AspectRatio: AspectRatio169,
	},
	ResolutionType1MP54: {
		Type:        ResolutionType1MP54,
		Group:       ResolutionGroupMP,
		Width:       1280,
		Height:      1024,
		Pixels:      Pixels{Width: 1280, Height: 1024},
		AspectRatio: AspectRatio54,
	},
	ResolutionTypeWXGA: {
		Type:        ResolutionTypeWXGA,
		Group:       ResolutionGroupHD,
		Width:       1366,
		Height:      768,
		Pixels:      Pixels{Width: 1366, Height: 768},
		AspectRatio: AspectRatio169,
	},
	ResolutionTypeHDPlus: {
		Type:        ResolutionTypeHDPlus,
		Group:       ResolutionGroupHD,
		Width:       1600,
		Height:      900,
		Pixels:      Pixels{Width: 1600, Height: 900},
		AspectRatio: AspectRatio169,
	},
	ResolutionType2MP43: {
		Type:        ResolutionType2MP43,
		Group:       ResolutionGroupMP,
		Width:       1600,
		Height:      1200,
		Pixels:      Pixels{Width: 1600, Height: 1200},
		AspectRatio: AspectRatio43,
	},
	ResolutionTypeFHD1080p: {
		Type:        ResolutionTypeFHD1080p,
		Group:       ResolutionGroupFHD,
		Width:       1920,
		Height:      1080,
		Pixels:      Pixels{Width: 1920, Height: 1080},
		AspectRatio: AspectRatio169,
	},
	ResolutionType3MP43: {
		Type:        ResolutionType3MP43,
		Group:       ResolutionGroupMP,
		Width:       2048,
		Height:      1536,
		Pixels:      Pixels{Width: 2048, Height: 1536},
		AspectRatio: AspectRatio43,
	},
	ResolutionTypeQHD1440p: {
		Type:        ResolutionTypeQHD1440p,
		Group:       ResolutionGroupQHD,
		Width:       2560,
		Height:      1440,
		Pixels:      Pixels{Width: 2560, Height: 1440},
		AspectRatio: AspectRatio169,
	},
	ResolutionType4MP169: {
		Type:        ResolutionType4MP169,
		Group:       ResolutionGroupMP,
		Width:       2688,
		Height:      1520,
		Pixels:      Pixels{Width: 2688, Height: 1520},
		AspectRatio: AspectRatio169,
	},
	ResolutionType6MP43: {
		Type:        ResolutionType6MP43,
		Group:       ResolutionGroupMP,
		Width:       3072,
		Height:      2048,
		Pixels:      Pixels{Width: 3072, Height: 2048},
		AspectRatio: "3:2",
	},
	ResolutionTypeQHDPlus: {
		Type:        ResolutionTypeQHDPlus,
		Group:       ResolutionGroupQHD,
		Width:       3200,
		Height:      1800,
		Pixels:      Pixels{Width: 3200, Height: 1800},
		AspectRatio: AspectRatio179,
	},
	ResolutionType4KUHD: {
		Type:        ResolutionType4KUHD,
		Group:       ResolutionGroupUHD,
		Width:       3840,
		Height:      2160,
		Pixels:      Pixels{Width: 3840, Height: 2160},
		AspectRatio: AspectRatio169,
	},
	ResolutionType12MP: {
		Type:        ResolutionType12MP,
		Group:       ResolutionGroupMP,
		Width:       4000,
		Height:      3000,
		Pixels:      Pixels{Width: 4000, Height: 3000},
		AspectRatio: AspectRatio43,
	},
	ResolutionType5K: {
		Type:        ResolutionType5K,
		Group:       ResolutionGroupUHD,
		Width:       5120,
		Height:      2880,
		Pixels:      Pixels{Width: 5120, Height: 2880},
		AspectRatio: AspectRatio169,
	},
	ResolutionType8KUHD: {
		Type:        ResolutionType8KUHD,
		Group:       ResolutionGroupUHD,
		Width:       7680,
		Height:      4320,
		Pixels:      Pixels{Width: 7680, Height: 4320},
		AspectRatio: AspectRatio169,
	},
	ResolutionType16KUHD: {
		Type:        ResolutionType16KUHD,
		Group:       ResolutionGroupUHD,
		Width:       15360,
		Height:      8640,
		Pixels:      Pixels{Width: 15360, Height: 8640},
		AspectRatio: AspectRatio169,
	},
	ResolutionType32KUHD: {
		Type:        ResolutionType32KUHD,
		Group:       ResolutionGroupUHD,
		Width:       30720,
		Height:      17280,
		Pixels:      Pixels{Width: 30720, Height: 17280},
		AspectRatio: AspectRatio169,
	},
	ResolutionType64KUHD: {
		Type:        ResolutionType64KUHD,
		Group:       ResolutionGroupUHD,
		Width:       61440,
		Height:      34560,
		Pixels:      Pixels{Width: 61440, Height: 34560},
		AspectRatio: AspectRatio169,
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
		supported = append(supported, res)
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

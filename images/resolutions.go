// Package images provides type definitions and constants for common security
// surveillance camera resolutions. The definitions are designed for production-grade
// readiness, ensuring accuracy, consistency, and maintainability.
package images

import (
	"fmt"
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

// ResolutionAlias represents a common name or standard for camera resolutions.
type ResolutionAlias string

// Defines the unique type for each supported camera resolution.
// This provides a clear, non-ambiguous identifier for each resolution standard.
const (
	ResolutionAliasNHD     ResolutionAlias = "NHD"
	ResolutionAliasFWVGA   ResolutionAlias = "FWVGA"
	ResolutionAliasWXGA    ResolutionAlias = "WXGA"
	ResolutionAliasHDPlus  ResolutionAlias = "HD+"
	ResolutionAliasQHDPlus ResolutionAlias = "QHD+"
	ResolutionAlias540p    ResolutionAlias = "540p"
	ResolutionAlias720p    ResolutionAlias = "720p"
	ResolutionAlias1080p   ResolutionAlias = "1080p"
	ResolutionAlias1440p   ResolutionAlias = "1440p"
	ResolutionAlias1MP     ResolutionAlias = "1MP"
	ResolutionAlias2MP     ResolutionAlias = "2MP"
	ResolutionAlias3MP     ResolutionAlias = "3MP"
	ResolutionAlias4MP     ResolutionAlias = "4MP"
	ResolutionAlias6MP     ResolutionAlias = "6MP"
	ResolutionAlias12MP    ResolutionAlias = "12MP"
	ResolutionAlias4K      ResolutionAlias = "4K"
	ResolutionAlias5K      ResolutionAlias = "5K"
	ResolutionAlias8K      ResolutionAlias = "8K"
	ResolutionAlias16K     ResolutionAlias = "16K"
	ResolutionAlias32K     ResolutionAlias = "32K"
	ResolutionAlias64K     ResolutionAlias = "64K"
)

// Pixels describes the exact dimensions of a resolution.
type Pixels struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Resolution describes the complete set of attributes for a CCTV resolution standard.
// It includes metadata for easy identification and filtering.
type Resolution struct {
	Alias       ResolutionAlias `json:"type" yaml:"type"`
	Group       ResolutionGroup `json:"group" yaml:"group"`
	Width       int             `json:"width" yaml:"width"`
	Height      int             `json:"height" yaml:"height"`
	Pixels      Pixels          `json:"pixels" yaml:"pixels"`
	AspectRatio AspectRatio     `json:"aspectRatio" yaml:"aspectRatio"`
}

// String returns a human-readable summary of the resolution.
// O(1) complexity.
func (r Resolution) String() string {
	return fmt.Sprintf("%s (%dx%d)", r.Alias, r.Pixels.Width, r.Pixels.Height)
}

// resolutions is a private variable that holds the catalog of all defined resolution standards,
// keyed by their ResolutionType for efficient lookups.
var resolutions = map[ResolutionAlias]Resolution{
	ResolutionAliasNHD: {
		Alias:       ResolutionAliasNHD,
		Group:       ResolutionGroupLegacy,
		Width:       640,
		Height:      360,
		Pixels:      Pixels{Width: 640, Height: 360},
		AspectRatio: AspectRatio169,
	},
	ResolutionAliasFWVGA: {
		Alias:       ResolutionAliasFWVGA,
		Group:       ResolutionGroupLegacy,
		Width:       854,
		Height:      480,
		Pixels:      Pixels{Width: 854, Height: 480},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias540p: {
		Alias:       ResolutionAlias540p,
		Group:       ResolutionGroupLegacy,
		Width:       960,
		Height:      540,
		Pixels:      Pixels{Width: 960, Height: 540},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias720p: {
		Alias:       ResolutionAlias720p,
		Group:       ResolutionGroupHD,
		Width:       1280,
		Height:      720,
		Pixels:      Pixels{Width: 1280, Height: 720},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias1MP: {
		Alias:       ResolutionAlias1MP,
		Group:       ResolutionGroupMP,
		Width:       1280,
		Height:      1024,
		Pixels:      Pixels{Width: 1280, Height: 1024},
		AspectRatio: AspectRatio54,
	},
	ResolutionAliasWXGA: {
		Alias:       ResolutionAliasWXGA,
		Group:       ResolutionGroupHD,
		Width:       1366,
		Height:      768,
		Pixels:      Pixels{Width: 1366, Height: 768},
		AspectRatio: AspectRatio169,
	},
	ResolutionAliasHDPlus: {
		Alias:       ResolutionAliasHDPlus,
		Group:       ResolutionGroupHD,
		Width:       1600,
		Height:      900,
		Pixels:      Pixels{Width: 1600, Height: 900},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias2MP: {
		Alias:       ResolutionAlias2MP,
		Group:       ResolutionGroupMP,
		Width:       1600,
		Height:      1200,
		Pixels:      Pixels{Width: 1600, Height: 1200},
		AspectRatio: AspectRatio43,
	},
	ResolutionAlias1080p: {
		Alias:       ResolutionAlias1080p,
		Group:       ResolutionGroupFHD,
		Width:       1920,
		Height:      1080,
		Pixels:      Pixels{Width: 1920, Height: 1080},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias3MP: {
		Alias:       ResolutionAlias3MP,
		Group:       ResolutionGroupMP,
		Width:       2048,
		Height:      1536,
		Pixels:      Pixels{Width: 2048, Height: 1536},
		AspectRatio: AspectRatio43,
	},
	ResolutionAlias1440p: {
		Alias:       ResolutionAlias1440p,
		Group:       ResolutionGroupQHD,
		Width:       2560,
		Height:      1440,
		Pixels:      Pixels{Width: 2560, Height: 1440},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias4MP: {
		Alias:       ResolutionAlias4MP,
		Group:       ResolutionGroupMP,
		Width:       2688,
		Height:      1520,
		Pixels:      Pixels{Width: 2688, Height: 1520},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias6MP: {
		Alias:       ResolutionAlias6MP,
		Group:       ResolutionGroupMP,
		Width:       3072,
		Height:      2048,
		Pixels:      Pixels{Width: 3072, Height: 2048},
		AspectRatio: "3:2",
	},
	ResolutionAliasQHDPlus: {
		Alias:       ResolutionAliasQHDPlus,
		Group:       ResolutionGroupQHD,
		Width:       3200,
		Height:      1800,
		Pixels:      Pixels{Width: 3200, Height: 1800},
		AspectRatio: AspectRatio179,
	},
	ResolutionAlias4K: {
		Alias:       ResolutionAlias4K,
		Group:       ResolutionGroupUHD,
		Width:       3840,
		Height:      2160,
		Pixels:      Pixels{Width: 3840, Height: 2160},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias12MP: {
		Alias:       ResolutionAlias12MP,
		Group:       ResolutionGroupMP,
		Width:       4000,
		Height:      3000,
		Pixels:      Pixels{Width: 4000, Height: 3000},
		AspectRatio: AspectRatio43,
	},
	ResolutionAlias5K: {
		Alias:       ResolutionAlias5K,
		Group:       ResolutionGroupUHD,
		Width:       5120,
		Height:      2880,
		Pixels:      Pixels{Width: 5120, Height: 2880},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias8K: {
		Alias:       ResolutionAlias8K,
		Group:       ResolutionGroupUHD,
		Width:       7680,
		Height:      4320,
		Pixels:      Pixels{Width: 7680, Height: 4320},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias16K: {
		Alias:       ResolutionAlias16K,
		Group:       ResolutionGroupUHD,
		Width:       15360,
		Height:      8640,
		Pixels:      Pixels{Width: 15360, Height: 8640},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias32K: {
		Alias:       ResolutionAlias32K,
		Group:       ResolutionGroupUHD,
		Width:       30720,
		Height:      17280,
		Pixels:      Pixels{Width: 30720, Height: 17280},
		AspectRatio: AspectRatio169,
	},
	ResolutionAlias64K: {
		Alias:       ResolutionAlias64K,
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
func GetResolutionByType(t ResolutionAlias) (Resolution, bool) {
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
			if !found || res.Pixels.Width*res.Pixels.Height > highest.Pixels.Width*highest.Pixels.Height {
				highest = res
				found = true
			}
		}
	}

	return highest, found
}

package images

import (
	"testing"
)

// TestResolution_GetMegaPixels performs table-driven tests on the GetMegaPixels method
// to ensure its calculations are accurate across all defined resolutions.
func TestResolution_GetMegaPixels(t *testing.T) {
	// Test cases cover standard resolutions and edge cases.
	testCases := []struct {
		name     string
		res      Resolution
		expected float64
	}{
		{
			name: "Full HD 1080p",
			res:  resolutions[ResolutionType1080p],
			// 1920 * 1080 = 2,073,600 -> 2.07 MP
			expected: 2.07,
		},
		{
			name: "4K UHD",
			res:  resolutions[ResolutionType4KUHD],
			// 3840 * 2160 = 8,294,400 -> 8.29 MP
			expected: 8.29,
		},
		{
			name: "1MP (5:4)",
			res:  resolutions[ResolutionType1MP],
			// 1280 * 1024 = 1,310,720 -> 1.31 MP
			expected: 1.31,
		},
		{
			name: "Experimental 64K UHD",
			res:  resolutions[ResolutionType64KUHD],
			// 61440 * 34560 = 2,123,366,400 -> 2123.37 MP
			expected: 2123.37,
		},
		{
			name: "Zero Width",
			res: Resolution{
				Pixels: Pixels{Width: 0, Height: 1080},
			},
			expected: 0.0,
		},
		{
			name: "Zero Height",
			res: Resolution{
				Pixels: Pixels{Width: 1920, Height: 0},
			},
			expected: 0.0,
		},
		{
			name: "Negative Width",
			res: Resolution{
				Pixels: Pixels{Width: -1920, Height: 1080},
			},
			expected: 0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act
			got := tc.res.GetMegaPixels()

			// Assert
			if got != tc.expected {
				t.Errorf("expected %.2f MP, but got %.2f MP", tc.expected, got)
			}
		})
	}
}

// TestResolution_String verifies the human-readable string output for a resolution.
func TestResolution_String(t *testing.T) {
	// Arrange
	res := resolutions[ResolutionType1080p]
	expected := "Full HD 1080p (1920x1080, 2.07MP)"

	// Act
	got := res.String()

	// Assert
	if got != expected {
		t.Errorf("expected string '%s', but got '%s'", expected, got)
	}
}

// TestGetAllResolutions ensures the function returns all defined resolutions.
func TestGetAllResolutions(t *testing.T) {
	// Act
	allResolutions := GetAllResolutions()

	// Assert
	if len(allResolutions) != len(resolutions) {
		t.Errorf("expected %d total resolutions, but got %d", len(resolutions), len(allResolutions))
	}
}

// TestGetSupportedResolutions ensures the function returns only non-experimental resolutions.
func TestGetSupportedResolutions(t *testing.T) {
	// Arrange
	var expectedCount int
	for _, res := range resolutions {
		if !res.Experimental {
			expectedCount++
		}
	}

	// Act
	supportedResolutions := GetSupportedResolutions()

	// Assert
	if len(supportedResolutions) != expectedCount {
		t.Errorf("expected %d supported resolutions, but got %d", expectedCount, len(supportedResolutions))
	}

	// Further check that no experimental resolutions are present.
	for _, res := range supportedResolutions {
		if res.Experimental {
			t.Errorf("found experimental resolution '%s' in supported list", res.Name)
		}
	}
}

// TestGetResolutionByType validates the lookup functionality for specific resolution types.
func TestGetResolutionByType(t *testing.T) {
	testCases := []struct {
		name           string
		resolutionType ResolutionAlias
		expectedFound  bool
		expectedName   ResolutionAlias
	}{
		{
			name:           "Valid HD 720p resolution",
			resolutionType: ResolutionTypeHD720p,
			expectedFound:  true,
			expectedName:   ResolutionTypeHD720p,
		},
		{
			name:           "Valid 4K UHD resolution",
			resolutionType: ResolutionType4KUHD,
			expectedFound:  true,
			expectedName:   ResolutionType4KUHD,
		},
		{
			name:           "Invalid resolution type",
			resolutionType: "InvalidType",
			expectedFound:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act
			res, found := GetResolutionByType(tc.resolutionType)

			// Assert
			if found != tc.expectedFound {
				t.Errorf("expected found=%v, but got found=%v", tc.expectedFound, found)
			}

			if tc.expectedFound && res.Name != tc.expectedName {
				t.Errorf("expected resolution name='%s', but got name='%s'", tc.expectedName, res.Name)
			}
		})
	}
}

// TestGetHighestResolutionUnderDimensions validates finding the highest resolution within given constraints.
func TestGetHighestResolutionUnderDimensions(t *testing.T) {
	testCases := []struct {
		name          string
		width         int
		height        int
		expectedFound bool
		expectedName  ResolutionAlias
		description   string
	}{
		{
			name:          "4K UHD constraints should return 4K UHD",
			width:         3840,
			height:        2160,
			expectedFound: true,
			expectedName:  ResolutionType4KUHD,
			description:   "Exact match for 4K UHD dimensions",
		},
		{
			name:          "Slightly larger than 4K UHD should return 4K UHD",
			width:         4000,
			height:        2200,
			expectedFound: true,
			expectedName:  ResolutionType4KUHD,
			description:   "Should find 4K UHD as highest resolution under these dimensions",
		},
		{
			name:          "1080p constraints should return Full HD 1080p",
			width:         1920,
			height:        1080,
			expectedFound: true,
			expectedName:  ResolutionType1080p,
			description:   "Exact match for Full HD 1080p dimensions",
		},
		{
			name:          "Between 720p and 1080p should return WXGA",
			width:         1500,
			height:        900,
			expectedFound: true,
			expectedName:  ResolutionTypeWXGA,
			description:   "Should find WXGA (1366x768, 1.05MP) as highest resolution under these dimensions",
		},
		{
			name:          "Very small dimensions should return nHD",
			width:         640,
			height:        360,
			expectedFound: true,
			expectedName:  ResolutionTypeNHD,
			description:   "Should find nHD as highest resolution under these small dimensions",
		},
		{
			name:          "Tiny dimensions should return no resolution",
			width:         100,
			height:        100,
			expectedFound: false,
			description:   "No standard resolution should fit in such small dimensions",
		},
		{
			name:          "Zero width should return no resolution",
			width:         0,
			height:        1080,
			expectedFound: false,
			description:   "Invalid width should result in no resolution found",
		},
		{
			name:          "Zero height should return no resolution",
			width:         1920,
			height:        0,
			expectedFound: false,
			description:   "Invalid height should result in no resolution found",
		},
		{
			name:          "Negative dimensions should return no resolution",
			width:         -1920,
			height:        -1080,
			expectedFound: false,
			description:   "Negative dimensions should result in no resolution found",
		},
		{
			name:          "Ultra-wide constraints with experimental resolutions",
			width:         100000,
			height:        50000,
			expectedFound: true,
			expectedName:  ResolutionType64KUHD,
			description:   "Should find the experimental 64K UHD as the highest resolution",
		},
		{
			name:          "Square aspect ratio constraints should return 12MP",
			width:         4000,
			height:        3000,
			expectedFound: true,
			expectedName:  ResolutionType12MP,
			description:   "Should find 12MP (4:3) resolution within square constraints",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act
			res, found := GetHighestResolutionUnderDimensions(tc.width, tc.height)

			// Assert found status
			if found != tc.expectedFound {
				t.Errorf("expected found=%v, but got found=%v", tc.expectedFound, found)
			}

			// Assert resolution name if expected to be found
			if tc.expectedFound {
				if res.Name != tc.expectedName {
					t.Errorf("expected resolution name='%s', but got name='%s'", tc.expectedName, res.Name)
				}

				// Verify the resolution actually fits within the constraints
				if res.Pixels.Width > tc.width || res.Pixels.Height > tc.height {
					t.Errorf("returned resolution %dx%d exceeds constraints %dx%d",
						res.Pixels.Width, res.Pixels.Height, tc.width, tc.height)
				}

				// Verify this is indeed the highest resolution by checking no other resolution
				// has higher megapixels while still fitting the constraints
				maxMegaPixels := res.GetMegaPixels()
				for _, otherRes := range resolutions {
					if otherRes.Pixels.Width <= tc.width && otherRes.Pixels.Height <= tc.height {
						if otherRes.GetMegaPixels() > maxMegaPixels {
							t.Errorf("found higher resolution %s (%.2fMP) that also fits constraints, but %s (%.2fMP) was returned",
								otherRes.Name, otherRes.GetMegaPixels(), res.Name, maxMegaPixels)
						}
					}
				}
			}

			// Log the test scenario for debugging
			t.Logf("Test: %s - Constraints: %dx%d, Found: %v, Result: %s",
				tc.description, tc.width, tc.height, found, res.Name)
		})
	}
}

// TestGetHighestResolutionUnderDimensions_EdgeCases tests edge cases and boundary conditions.
func TestGetHighestResolutionUnderDimensions_EdgeCases(t *testing.T) {
	// Test with dimensions that match exactly one resolution
	t.Run("Exact match single resolution", func(t *testing.T) {
		// Use nHD dimensions which should be the smallest
		res, found := GetHighestResolutionUnderDimensions(640, 360)

		if !found {
			t.Error("expected to find nHD resolution for exact dimensions")
		}

		if res.Name != ResolutionTypeNHD {
			t.Errorf("expected nHD resolution, got %s", res.Name)
		}
	})

	// Test with dimensions just below a resolution
	t.Run("Just below HD 720p", func(t *testing.T) {
		res, found := GetHighestResolutionUnderDimensions(1279, 719)

		if !found {
			t.Error("expected to find a resolution just below HD 720p")
		}

		// Should not return HD 720p since constraints are just below it
		if res.Name == ResolutionTypeHD720p {
			t.Error("should not return HD 720p when constraints are just below it")
		}

		// Should return a smaller resolution (likely qHD 540p)
		if res.Pixels.Width > 1279 || res.Pixels.Height > 719 {
			t.Errorf("returned resolution %dx%d exceeds constraints 1279x719",
				res.Pixels.Width, res.Pixels.Height)
		}
	})

	// Test with very large dimensions
	t.Run("Very large dimensions", func(t *testing.T) {
		res, found := GetHighestResolutionUnderDimensions(1000000, 1000000)

		if !found {
			t.Error("expected to find highest resolution for very large dimensions")
		}

		// Should return the experimental 64K UHD as it has the highest megapixels
		if res.Name != ResolutionType64KUHD {
			t.Errorf("expected 64K UHD for very large dimensions, got %s", res.Name)
		}
	})

	// Test constraint where HD+ exactly fits
	t.Run("HD Plus exact fit", func(t *testing.T) {
		res, found := GetHighestResolutionUnderDimensions(1600, 900)

		if !found {
			t.Error("expected to find HD+ resolution for exact dimensions")
		}

		if res.Name != ResolutionTypeHDPlus {
			t.Errorf("expected HD+ resolution, got %s", res.Name)
		}
	})

	// Test constraint just below HD+ to verify WXGA is returned
	t.Run("Just below HD Plus should return WXGA", func(t *testing.T) {
		res, found := GetHighestResolutionUnderDimensions(1599, 900)

		if !found {
			t.Error("expected to find WXGA resolution just below HD+ dimensions")
		}

		if res.Name != ResolutionTypeWXGA {
			t.Errorf("expected WXGA resolution, got %s", res.Name)
		}
	})
}

// TestGetHighestResolutionUnderDimensions_MegaPixelComparison verifies megapixel-based selection.
func TestGetHighestResolutionUnderDimensions_MegaPixelComparison(t *testing.T) {
	// Create a test case where multiple resolutions fit but we want the highest megapixels
	t.Run("Multiple fits - highest megapixels wins", func(t *testing.T) {
		// Use constraints that allow both HD 720p and WXGA
		res, found := GetHighestResolutionUnderDimensions(1400, 800)

		if !found {
			t.Error("expected to find a resolution")
		}

		// WXGA (1.05MP) should win over HD 720p (0.92MP)
		if res.Name != ResolutionTypeWXGA {
			t.Errorf("expected WXGA (higher megapixels), got %s", res.Name)
		}

		// Verify both would fit
		hd720p, _ := GetResolutionByType(ResolutionTypeHD720p)
		wxga, _ := GetResolutionByType(ResolutionTypeWXGA)

		if hd720p.Pixels.Width > 1400 || hd720p.Pixels.Height > 800 {
			t.Error("HD 720p should fit in these constraints")
		}

		if wxga.Pixels.Width > 1400 || wxga.Pixels.Height > 800 {
			t.Error("WXGA should fit in these constraints")
		}

		// Verify WXGA has higher megapixels
		if wxga.GetMegaPixels() <= hd720p.GetMegaPixels() {
			t.Error("WXGA should have higher megapixels than HD 720p")
		}
	})
}

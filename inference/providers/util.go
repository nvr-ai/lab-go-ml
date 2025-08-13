// Package inference - Utility functions.
package providers

import "runtime"

// GetSharedLibPath returns the path to the shared library for the current platform.
//
// Returns:
//   - string: The path to the shared library.
func GetSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/libonnxruntime.1.23.0.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "./third_party/libonnxruntime.1.23.0.dylib"
		}

	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so"
		}
		return "../third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

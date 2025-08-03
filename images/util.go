package images

import (
	"crypto/md5"
	"fmt"

	"gocv.io/x/gocv"
)

// ComputeMatChecksum generates a deterministic checksum for a Mat to verify idempotency.
//
// Arguments:
// - mat: The Mat to compute checksum for.
//
// Returns:
// - A hex-encoded MD5 checksum string.
//
// Example:
//
// ```go
//
//	checksum := ComputeMatChecksum(frame)
//	fmt.Printf("Frame checksum: %s\n", checksum)
//
// ```
func ComputeMatChecksum(mat gocv.Mat) string {
	if mat.Empty() {
		return "empty"
	}

	data, _ := mat.DataPtrUint8()
	hash := md5.New()
	hash.Write(data)
	return fmt.Sprintf("%x", hash.Sum(nil))
}

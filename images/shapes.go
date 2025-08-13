// Package images - Image processing utilities
package images

// Rect is a lightweight bounding box.
type Rect struct {
	// X2,Y2 are exclusive (like image.Rectangle).
	X1 float32 `json:"x1" yaml:"x1"`
	Y1 float32 `json:"y1" yaml:"y1"`
	X2 float32 `json:"x2" yaml:"x2"`
	Y2 float32 `json:"y2" yaml:"y2"`
}

// CalculateIoU calculates the Intersection over Union (IoU) between two rectangles.
//
// Arguments:
//   - x1: The x-coordinate of the first rectangle.
//   - y1: The y-coordinate of the first rectangle.
//   - x2: The x-coordinate of the second rectangle.
//   - y2: The y-coordinate of the second rectangle.
//   - x3: The x-coordinate of the third rectangle.
//   - y3: The y-coordinate of the third rectangle.
//   - x4: The x-coordinate of the fourth rectangle.
//   - y4: The y-coordinate of the fourth rectangle.
//
// Returns:
//   - float32: The IoU score between the two rectangles.
//
// Example Usage:
// ```go
// rect1 := Rect{X1: 0, Y1: 0, X2: 10, Y2: 10}
// rect2 := Rect{X1: 5, Y1: 5, X2: 15, Y2: 15}
// iou := CalculateIoU(rect1, rect2)
// fmt.Println(iou) // Output: 0.142857
// ```
//
// See also:
//   - http://ronny.rest/tutorials/module/localization_001/iou
//
// IoU (Intersection over Union) is a fundamental metric in computer vision,
// particularly in object detection, that measures the extent of overlap between
// two bounding boxes (rectangles). This method calculates that value.
//
// See also:
//   - http://ronny.rest/tutorials/module/localization_001/iou
//
// **The Core Concept: What is IoU?**
//
// Imagine you have two rectangular areas. The IoU is a number between 0.0 and 1.0
// that answers the question: "How much do these two rectangles overlap?"
//
// It is formally defined by the formula:
//
//	IoU = Area of Intersection / Area of Union
//
//	- A value of 1.0 means the rectangles are identical; they overlap perfectly.
//	- A value of 0.0 means the rectangles don't overlap at all.
//	- A value of 0.5 means the area where they intersect is half the size of the total area they
//
// cover combined.
//
// This metric is crucial for evaluating the accuracy of an object detection model.
// For example, if a model predicts a bounding box for a "car", we can compare it
// to the ground-truth (the "correct") bounding box using IoU. If the IoU is
// above a certain threshold (e.g., 0.7), we consider the detection a "hit".
//
// **How This Method Implements the Calculation**
//
// The method is attached to a `Rect` type, which we call the "receiver" (`r`). It
// takes another `Rect` (`o`) as an argument and returns the `float32` IoU score.
// The calculation is broken down into three main steps:
//
// **1. Calculate the Intersection Area**
//
//	This is the most nuanced part. To find the area of the overlapping rectangle,
//	we first need to find its coordinates.
//
//	- The top-left corner of the intersection (`ix1`, `iy1`) is found by taking
//	the *maximum* of the top-left corners of the two input rectangles.
//	Think about it: the overlap can't start before *both* rectangles have begun.
//
//	- The bottom-right corner of the intersection (`ix2`, `iy2`) is found by taking
//	the *minimum* of the bottom-right corners. Similarly, the overlap
//	must end as soon as the *first* rectangle ends.
//
//	- An essential edge case is handled here: If the rectangles do not overlap,
//	the calculated width (`ix2 - ix1`) or height (`iy2 - iy1`) will be
//	zero or negative. In this scenario, the intersection area is zero, and we
//	correctly return 0.0 immediately, avoiding division by zero or non-sensical results.
//
// **2. Calculate the Union Area**
//
//	The "Union" is the total area covered by both rectangles combined. A naive
//	approach would be to simply add their areas (`areaR + areaO`). However, this
//	would double-count the overlapping (intersection) area.
//
//	We use the "Principle of Inclusion-Exclusion" to correct for this:
//
//	  Area(Union) = Area(A) + Area(B) - Area(Intersection)
//
//	  This method first calculates the individual areas of `r` and `o`, then
//	  subtracts the intersection area we already found in step 1.
//
// **3. Divide and Return**
//
//	Finally, the method performs the division `intersection / union`. It's critical
//	that we cast these values to `float32` before the division. If we used
//	integer division, the result would almost always be 0 (since the intersection
//	is smaller than the union), which would be incorrect.
//
// Arguments:
//   - r (receiver Rect): The first rectangle, on which the method is called.
//   - o (Rect): The other rectangle to compare against.
//
// Returns:
//   - float32: A value between 0.0 and 1.0 representing the IoU score.
//
// Example Usage:
// ```go
//
//	rect1 := Rect{X1: 0, Y1: 0, X2: 10, Y2: 10} // Define two rectangles that have some overlap.
//	rect2 := Rect{X1: 5, Y1: 5, X2: 15, Y2: 15}
//
//	iouScore := rect1.IoU(rect2) // Calculate the IoU score.
//
//	fmt.Printf("The IoU is: %f\n", iouScore) // intersection is a 5x5 box (area=25), union is 100 +
//
// 100 - 25 = 175. Output: The IoU is: 0.142857
//
// ```
func CalculateIoU(x1, y1, x2, y2, x3, y3, x4, y4 float32) float32 {
	// First, calculate the coordinates of the intersection rectangle.
	//
	// The intersection is the overlapping area of the two rectangles.
	// Its coordinates are found by taking the maximum of the starting coordinates
	// and the minimum of the ending coordinates.
	ix1 := MaxFloat32(x1, x3)
	iy1 := MaxFloat32(y1, y3)
	ix2 := MinFloat32(x2, x4)
	iy2 := MinFloat32(y2, y4)

	// Second, calculate the intersection area, handling non-overlapping cases.
	// Calculate the width and height of the intersection. If either is zero or
	// negative, the rectangles do not overlap, and the intersection area is 0.
	interW := ix2 - ix1
	interH := iy2 - iy1
	if interW <= 0 || interH <= 0 {
		return 0.0
	}
	interArea := interW * interH

	// Third, calculate the union area.
	// We use the Principle of Inclusion-Exclusion:
	// Union(A, B) = Area(A) + Area(B) - Intersection(A, B)
	areaR := (x2 - x1) * (y2 - y1)
	areaO := (x4 - x3) * (y4 - y3)
	unionArea := areaR + areaO - interArea

	// Finally, compute and return the final IoU score.
	// Cast to float32 to ensure floating-point division.
	return interArea / unionArea
}

// MaxFloat32 returns the maximum of two float32 values.
//
// Arguments:
//   - a: First value.
//   - b: Second value.
//
// Returns:
//   - The larger of the two values.
func MaxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// MinFloat32 returns the minimum of two float32 values.
//
// Arguments:
//   - a: First value.
//   - b: Second value.
//
// Returns:
//   - The smaller of the two values.
func MinFloat32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

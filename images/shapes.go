// Package images - Image processing utilities
package images

// Rect is a lightweight bounding box.
type Rect struct {
	// X2,Y2 are exclusive (like image.Rectangle).
	X1, Y1, X2, Y2 int
}

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
//	- A value of 0.5 means the area where they intersect is half the size of the total area they cover combined.
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
//	fmt.Printf("The IoU is: %f\n", iouScore) // intersection is a 5x5 box (area=25), union is 100 + 100 - 25 = 175. Output: The IoU is: 0.142857
//
// ```
func CalculateIoU(r, o Rect) float32 {
	// First, calculate the coordinates of the intersection rectangle.
	//
	// The intersection is the overlapping area of the two rectangles.
	// Its coordinates are found by taking the maximum of the starting coordinates
	// and the minimum of the ending coordinates.
	ix1 := max(r.X1, o.X1)
	iy1 := max(r.Y1, o.Y1)
	ix2 := min(r.X2, o.X2)
	iy2 := min(r.Y2, o.Y2)

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
	areaR := (r.X2 - r.X1) * (r.Y2 - r.Y1)
	areaO := (o.X2 - o.X1) * (o.Y2 - o.Y1)
	unionArea := areaR + areaO - interArea

	// Finally, compute and return the final IoU score.
	// Cast to float32 to ensure floating-point division.
	return float32(interArea) / float32(unionArea)
}

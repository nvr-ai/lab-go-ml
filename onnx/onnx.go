package onnx

// import (
// 	"image"

// 	ort "github.com/yalue/onnxruntime_go"
// )

// func main() {
// 	ort.SetSharedLibraryPath("onnxruntime.so")
// 	ort.InitializeEnvironment()

// 	session, _ := ort.NewAdvancedSession("yolov8.onnx")
// 	inputTensor := ort.NewTensorFromImage("input.jpg", image.Pt(640, 640))
// 	outputTensor := ort.NewEmptyTensor[float32](image.Pt(640, 640))

// 	session.Run([]*Tensor{inputTensor}, []*Tensor{outputTensor})

// 	boxes := DecodeYOLOOutput(outputTensor)
// 	DrawBoxes("input.jpg", boxes)
// }

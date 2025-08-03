ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24
export 

tidy:
	go mod tidy

gocv/webcam: tidy
	go run ./cmd/gocv/main.go


gorgonia/tiny-yolo-v3-coco: tidy
	cd gorgonia/tiny-yolo-v3-coco && go run .
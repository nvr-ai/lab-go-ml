import torch
import onnxruntime as ort
import numpy as np
import cv2
from torchvision import transforms

# Load PyTorch model
from repo.src.zoo.dfine.dfine_decoder import build_dfine_model  # Adjust to your actual import
model = build_dfine_model()
model.load_state_dict(torch.load("configs/dfine/dfine_hgnetv2_l_coco.yml"))
model.eval()

# Load and preprocess image
def preprocess_image(img_path, size=(640, 640)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, size)
    tensor = transforms.ToTensor()(img_resized)
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])(tensor)
    return tensor.unsqueeze(0), img_resized

img_tensor, img_np = preprocess_image("../../../../ml/corpus/images/videos/freeway-view-22-seconds-1080p.mp4/frame-0.jpg")

# Run PyTorch inference
with torch.no_grad():
    pt_output = model(img_tensor)
    pt_boxes = pt_output["boxes"].cpu().numpy()
    pt_scores = pt_output["scores"].cpu().numpy()

# Run ONNX inference
ort_session = ort.InferenceSession("repo/model.onnx")
ort_inputs = {"images": img_tensor.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_boxes = ort_outputs[0]
onnx_scores = ort_outputs[1]

# Compare outputs
def compare_outputs(pt_boxes, onnx_boxes, pt_scores, onnx_scores, iou_threshold=0.5):
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    matches = []
    for i, pt_box in enumerate(pt_boxes):
        for j, onnx_box in enumerate(onnx_boxes):
            if iou(pt_box, onnx_box) > iou_threshold:
                matches.append((i, j, pt_scores[i], onnx_scores[j]))
                break

    print(f"Matched {len(matches)} boxes with IOU > {iou_threshold}")
    for i, j, pt_score, onnx_score in matches:
        print(f"Box {i}: PyTorch score={pt_score:.3f}, ONNX score={onnx_score:.3f}")

compare_outputs(pt_boxes, onnx_boxes, pt_scores, onnx_scores)

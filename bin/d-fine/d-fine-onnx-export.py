#!/usr/bin/env uv run

import torch
import torchvision.transforms as transforms
from PIL import Image
import onnx

# 1. Load the D-FINE model (replace with actual loading code from repository)
# Note: You must implement this part based on D-FINE's original implementation
class D_FINE(torch.nn.Module):
    def __init__(self):
        super(D_FINE, self).__init__()
        # Replace with actual model architecture

    def forward(self, x):
        # Replace with actual model logic
        return x

model = D_FINE()
model.load_state_dict(torch.load("path/to/pretrained_weights.pth"))  # Replace with actual weights
model.eval()

# 2. Prepare dummy input (adjust shape based on D-FINE's requirements)
# Assume input is 3-channel image of size 256x256
dummy_input = torch.randn(1, 3, 256, 256)  # batch_size=1, channels=3, height=256, width=256

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "d_fine.onnx",
    export_params=True,          # Store trained parameter values
    opset_version=13,           # ONNX opset version (choose latest)
    do_constant_folding=True,   # Optimize constant folding
    input_names=["input"],      # Input name in ONNX graph
    output_names=["output"],    # Output name in ONNX graph
    verbose=True,  # Print detailed export information
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"}
    },  # Enable dynamic batch size and image dimensions
    verbose=True  # Print detailed export information
)

print("Model exported to d_fine.onnx")
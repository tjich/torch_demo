import torch
import torch.nn as nn
import torch.onnx 
import numpy as np


class MyNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10)
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

device = "cpu"

model_name = "first_model_0709.pt"
model_loaded = MyNN().to(device)
model_loaded.load_state_dict(torch.load(model_name))
model_loaded.eval()

onnx_model_name = "first_demo_0709.onnx"
batch_size = 32
x = torch.randn(batch_size, 1, 28, 28)
torch.onnx.export(model_loaded,              # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_model_name,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
print(f"Onnx model save: {onnx_model_name}")
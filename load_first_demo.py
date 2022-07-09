from turtle import forward
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

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

model_name = "first_model_0709.pt"
model_loaded = MyNN().to("cpu")
model_loaded.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=False,
  transform=transforms.ToTensor()
)

batch_size = 1
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
  pred = model_loaded(x)
  pred_label = pred.argmax(dim=1)
  print(f"pred_label: {pred_label}, y: {y}")
  break
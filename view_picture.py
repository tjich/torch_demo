import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=False,
  transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=False,
  transform=transforms.ToTensor()
)

batch_size = 8
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

fig = plt.figure

for x, y in test_dataloader:
  print(f"Shape of x: {x.shape}")
  print(f"Shape of y: {y.shape}")
  image = x[0].numpy().reshape(28, 28)
  plt.imshow(image, cmap="gray")
  plt.show()
  break
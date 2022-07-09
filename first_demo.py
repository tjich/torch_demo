from turtle import forward
import torch
from torch import nn
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

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for x, y in test_dataloader:
#   print(f"Shape of x: {x.shape}")
#   print(f"Shape of y: {y.shape}")
#   break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

model = MyNN().to(device)
# print(model)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

print_step = 100

def train(dataloader, model, loss_func, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch % print_step == 0:
      loss, current = loss.item(), batch * len(x)
      print(f"loss: {loss:>7f} [{current:>5d} / size: {size:>5d}]")

def test(dataloader, model, loss_func):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for x, y in dataloader:
      x, y = x.to(device), y.to(device)
      pred = model(x)
      test_loss += loss_func(pred, y).item()
      correct += (pred.argmax(dim=1) == y).sum().item()
  correct /= size
  test_loss /= num_batches
  print(f"Test Error:\n Acc: {(100*correct):>0.1f}%, Avg loss: {test_loss:.8f}\n")

epochs = 50
for t in range(epochs):
  print(f"Epoch {t+1}\n----------------")
  train(train_dataloader, model, loss_func, optimizer)
  test(test_dataloader, model, loss_func)
print("Done!")

model_name = "first_model_0709.pt"
torch.save(model.state_dict(), model_name)
print(f"Saved model: {model_name}")

model_loaded = MyNN().to(device)
model_loaded.load_state_dict(torch.load(model_name))
model_loaded.eval()
classes = [
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot"
]

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
  pred = model(x)
  predicted, actual = classes[pred[0].argmax(0)], classes[y]
  print(f'Predicted: "{predicted}", Actual: "{actual}"')


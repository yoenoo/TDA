import torch.nn as nn

# model
class LRNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(28*28,1,bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc(x)
    return self.sigmoid(x)
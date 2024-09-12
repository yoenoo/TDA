import torch.nn as nn

class LRNet(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.fc = nn.Linear(input_dim, 1, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.fc(x)
    x = self.sigmoid(x)
    return x
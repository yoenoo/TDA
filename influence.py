import numpy as np 
from tqdm import tqdm, trange
# from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# from tda.base import BaseInfluenceModule
from tda.modules import AutogradInfluenceModule


# MNIST example
BATCH_SIZE = 128

transform = transforms.Compose([
  transforms.ToTensor(),
])

pos_class = 1
neg_class = 7

# train data
mnist_train = datasets.MNIST(root="./data", train=True, 
                             download=True, transform=transform)
images, labels = mnist_train.data, mnist_train.targets

labels_idx = torch.where(torch.isin(labels, torch.tensor([pos_class, neg_class])))
images = images[labels_idx] / 255. # normalize to (0,1)
labels = labels[labels_idx]

ds = TensorDataset(images, labels)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE)

# test data
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
images, labels = mnist_test.data, mnist_test.targets

labels_idx = torch.where(torch.isin(labels, torch.tensor([pos_class, neg_class])))
images = images[labels_idx] / 255. # normalize to (0,1)
labels = labels[labels_idx]

ds = TensorDataset(images, labels)
test_loader = DataLoader(ds, batch_size=BATCH_SIZE) 


class LogisticRegression(nn.Module):
  def __init__(self, f_in, f_out=1):
    super().__init__()
    self.fc = nn.Linear(f_in, f_out)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.fc(x)
    x = self.sigmoid(x)
    return x


model = LogisticRegression(28*28)
optimizer = SGD(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

epochs = 50

# -- training
model.train()
for epoch in (pbar := trange(epochs)):
  batch_losses = []
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()

    data = data.reshape(-1, 28*28)
    target = torch.where(target == neg_class, 1., 0.).reshape(-1,1)
    output = model(data)
    
    loss = F.binary_cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    
    # output = model(data.reshape(-1,28*28))
    # target_preds = torch.where(output < 0.5, 1, 7).reshape(-1)
    # acc = (target == target_preds).sum() / len(output)
    
    batch_losses.append(loss.item())
      
  epoch_loss = np.mean(batch_losses)
  pbar.set_description(f"Epoch: {epoch}, Average loss: {epoch_loss:.4f}")

print(model)

# -- post-training
module = AutogradInfluenceModule(model, train_loader, test_loader, damp=0.1)

train_idxs = list(range(len(train_loader.dataset)))
test_idxs = [3]
influences = module.influences(train_idxs=train_idxs, test_idxs=test_idxs)
print(influences)

# harmful iamge
x_test, y_test = test_loader.dataset.tensors
x_test = x_test[test_idxs]
y_test = y_test[test_idxs]

flipped_idx = train_loader.dataset.tensors[1] != y_test

# finding a training image that has the same label but is harmful
train_images = train_loader.dataset.tensors[0]
train_labels = train_loader.dataset.tensors[1]

influence_orders = torch.argsort(influences[~flipped_idx])

train_idx = torch.where(~flipped_idx)[0][influence_orders[0]]
print(train_idx, train_labels[train_idx])

# harmful_train_image = train_loader.dataset[influence_orders[0]][0]
harmful_train_image = train_images[train_idx]

import matplotlib.pyplot as plt 
_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(x_test.reshape(28,28), cmap="Greys")
ax1.set_title(f"Label={y_test}")
ax2.imshow(harmful_train_image, cmap="Greys")
plt.show()
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

# TODO: cleaner data processing
# train/val data
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
test_images = images[labels_idx] / 255. # normalize to (0,1)
test_labels = labels[labels_idx]

ds = TensorDataset(test_images, test_labels)
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

# -- pre-training
train_idxs = list(range(len(train_loader.dataset)))
module = AutogradInfluenceModule(model, train_loader, test_loader, damp=0.01)
test_idxs = [384] # test example with minimal loss
influences = module.influences(train_idxs=train_idxs, test_idxs=test_idxs) * len(train_idxs)

sorted_influences = torch.sort(influences).values
influence_gaps = []
for i in range(1, len(sorted_influences)):
  delta = sorted_influences[i] - sorted_influences[i-1]
  influence_gaps.append(delta)

print(np.std(influence_gaps))
print(torch.min(influences), torch.max(influences), torch.std(influences))

# import matplotlib.pyplot as plt 
# plt.hist(influence_gaps, density=True)
# plt.show()


model = LogisticRegression(28*28)
optimizer = SGD(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
epochs = 500

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


x_test = test_images.reshape(-1, 28*28)
y_test = torch.where(test_labels == neg_class, 1., 0.).reshape(-1,1)
test_losses = F.binary_cross_entropy(model(x_test), y_test, reduction="none")
test_idxs = torch.argsort(test_losses, dim=0, descending=True)[:5]
# print(test_idxs)

test_idxs = torch.argsort(test_losses, dim=0)[:5]
# print(test_idxs)

train_idxs = list(range(len(train_loader.dataset)))

# -- post-training
module = AutogradInfluenceModule(model, train_loader, test_loader, damp=0.01)
test_idxs = [384] # test example with minimal loss
# test_idxs = [test_idxs[0]] # test example with minimal loss
influences = module.influences(train_idxs=train_idxs, test_idxs=test_idxs) * len(train_idxs)

sorted_influences = torch.sort(influences).values
influence_gaps = []
for i in range(1, len(sorted_influences)):
  delta = sorted_influences[i] - sorted_influences[i-1]
  influence_gaps.append(delta)

print(np.std(influence_gaps))
print(torch.min(influences), torch.max(influences), torch.std(influences))

import matplotlib.pyplot as plt 
plt.hist(influence_gaps, density=True)
plt.show()

# # distribution of influence values
# import matplotlib.pyplot as plt 
# plt.hist(influences, bins=50)
# plt.show()
# exit()

# # TODO
# # split conformal prediction
# n = len(influences)
# alpha = 0.1
# scores = influences
# q_level = np.ceil((n+1)*(1-alpha))/n
# qhat = np.quantile(scores, q_level, method="higher")
# val_smx = module.influences(train_idxs=train_idxs, test_idxs=[4, 5])
# print(val_smx.shape)
# prediction_sets = val_smx <= qhat
# print(len(prediction_sets), n)
# # print(train_idxs[prediction_sets])

# harmful iamge
x_test, y_test = test_loader.dataset.tensors
x_test = x_test[test_idxs]
y_test = y_test[test_idxs]

train_images = train_loader.dataset.tensors[0]
train_labels = train_loader.dataset.tensors[1]

influence_orders = torch.argsort(influences)

train_idx_top_5 = influence_orders[:5] # harmful (i.e. smallest 5)
train_idx_bottom_5 = influence_orders[-5:] # helpful (i.e. largest 5)

import matplotlib.pyplot as plt 
_, ax = plt.subplots(ncols=5, nrows=2)
for i, idx in enumerate(train_idx_top_5):
  ax[0][i].imshow(train_images[idx], cmap="Greys")
  label = train_labels[idx].item()
  _if = round(influences[idx].item(), 5)
  ax[0][i].set_title(f"label={label}, IF={_if}", fontsize=6)

for i, idx in enumerate(train_idx_bottom_5):
  ax[1][i].imshow(train_images[idx], cmap="Greys")
  label = train_labels[idx].item()
  _if = round(influences[idx].item(), 5)
  ax[1][i].set_title(f"label={label}, IF={_if}", fontsize=6)

plt.tight_layout()
plt.show()

exit()


flipped_idx = train_loader.dataset.tensors[1] != y_test

# finding a training image that has the same label but is harmful
train_images = train_loader.dataset.tensors[0]
train_labels = train_loader.dataset.tensors[1]

influence_orders = torch.argsort(influences[~flipped_idx])

train_idx_top_5 = torch.where(~flipped_idx)[0][influence_orders[:5]]
train_idx_bottom_5 = torch.where(~flipped_idx)[0][influence_orders[-5:]]
# print(train_idx, train_labels[train_idx])

# harmful_train_image = train_images[train_idx]
# good_train_image = train_images[train_idx_g]

import matplotlib.pyplot as plt 
_, ax = plt.subplots(ncols=5, nrows=2)
# ax1.imshow(x_test.reshape(28,28), cmap="Greys")
# ax1.set_title(f"Label={y_test}")
for i, idx in enumerate(train_idx_top_5):
  ax[0][i].imshow(train_images[idx], cmap="Greys")

for i, idx in enumerate(train_idx_bottom_5):
  ax[1][i].imshow(train_images[idx], cmap="Greys")

# ax2.imshow(harmful_train_image, cmap="Greys")
# ax3.imshow(good_train_image, cmap="Greys")
plt.tight_layout()
plt.show()
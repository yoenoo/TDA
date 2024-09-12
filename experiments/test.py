import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

from datautils import load_mnist
from models import LRNet 
# from tda.modules import AutogradInfluenceModule

pos_class = 0
neg_class = 4
damp = 1e-2

train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=(pos_class, neg_class))

X_train, Y_train = train_loader.dataset.tensors
print(X_train.shape, Y_train.shape)

# model = LRNet(28*28)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(fit_intercept=False, penalty="l2", C=damp)
model = model.fit(X_train, Y_train)

# recreate model in PyTorch
fc = nn.Linear(28*28, 1, bias=False)
#fc.weight = nn.Parameter(torch.tensor(model.coef_).to(dtype=torch.float32))

model = nn.Sequential()
model.add_module("fc", fc)
model.add_module("sigmoid", nn.Sigmoid())

print(model)

def compute_cost(xs, ys, params, damp=None, elem=False):
  yhat = model(xs.reshape(-1, 28*28))
  ys = torch.where(ys == neg_class, 1., 0.).reshape(-1,1)
  cost_elem = -torch.sum(ys*torch.log(yhat) + (1-ys)*torch.log(1-yhat), dim=1)
  if elem: return cost_elem
  cost = torch.mean(cost_elem)
  if damp is not None:
    # w = model.fc.weight
    # cost += damp * w.pow(2).sum().sqrt() # L2 loss
    cost += damp * params.pow(2).sum().sqrt()
  return cost

def compute_acc(xs, ys):
  ys_pred = model(xs.reshape(-1, 28*28))
  ys_pred_label = torch.where(ys_pred < 0.5, pos_class, neg_class).reshape(-1)
  acc = (ys_pred_label == ys).sum() / len(ys)
  return acc


# # uncomment for neural network training
# epochs = 0
# # optimizer = Adam(model.parameters(), lr=1e-2)
# optimizer = SGD(model.parameters(), lr=1e-4)
# model.train()
# for epoch in trange(epochs):
#   for xs, ys in train_loader:
#     optimizer.zero_grad()

#     loss = compute_cost(xs, ys, params=model.fc.weight, damp=damp)
#     loss.backward()
#     optimizer.step()
    
#   if (epoch % 100 == 0) and (epoch > 0):
#     x_train, y_train = train_loader.dataset.tensors
#     acc_train = compute_acc(x_train, y_train)
#     x_test, y_test = test_loader.dataset.tensors
#     acc_test = compute_acc(x_test, y_test)
#     cost_test = compute_cost(x_test, y_test, params=model.fc.weight, damp=damp)
#     print(acc_train.item(), acc_test.item(), cost_test.item())

# calculate influence funcitons
x_train, y_train = train_loader.dataset.tensors
def f(theta_):
  return compute_cost(x_train, y_train, theta_, damp=damp)

hessian = torch.autograd.functional.hessian(f, model.fc.weight.reshape(-1))
inv_hessian = torch.inverse(hessian)

x_test, y_test = test_loader.dataset.tensors

test_cost_elem = compute_cost(x_test, y_test, model.fc.weight, damp=damp, elem=True)
# test_idx = [torch.argmin(test_cost_elem)]
test_idx = [1778]
print(test_idx)

test_cost = compute_cost(x_test[test_idx], y_test[test_idx], model.fc.weight, damp=damp)
print(test_cost)
test_grad = torch.autograd.grad(test_cost, model.fc.weight)

ivhp = test_grad[0] @ inv_hessian # s_test

x_train, y_train = train_loader.dataset.tensors
scores = []
for i in trange(len(train_loader)):
  train_cost = compute_cost(x_train[i], y_train[i], model.fc.weight, damp=damp)
  train_grad = torch.autograd.grad(train_cost, model.fc.weight)
  _if = train_grad[0] @ ivhp.T
  _if /= len(x_train)
  # _if = ivhp @ train_grad[0] # or: np.concatenate or somehting like that 
  
  # if y_train[i] != neg_class:
  #   continue
  scores.append((i, _if))

scores = sorted(scores, key=lambda x: x[1]) # top scores = harmful, bottom scores = helpful

scores_only = [s[1].item() for s in scores]

scores_gaps = [scores_only[i]-scores_only[i-1] for i in range(1,len(scores_only))]
_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.hist(scores_only, density=True)
ax2.hist(scores_gaps, density=True)




_, ax = plt.subplots(nrows=2, ncols=7, figsize=(16,4))
ax[0][0].imshow(x_test[test_idx].reshape(28,28), cmap="gray", interpolation="nearest")
label = y_test[test_idx].item()
ax[0][0].set_title(f"Test image ({label})", fontsize=8)
ax[0][0].axis("off")
for i in range(1, 7):
  ax[0][i].imshow(x_train[scores[i-1][0]].reshape(28,28), cmap="gray", interpolation="nearest")
  label = y_train[scores[i-1][0]]
  if_score = scores[i-1][1].item()
  ax[0][i].set_title(f"Harmful image {scores[i-1][0]} ({label})\nIF={round(if_score,3)}", fontsize=6)
  ax[0][i].axis("off")

ax[1][0].imshow(x_test[test_idx].reshape(28,28), cmap="gray", interpolation="nearest")
label = y_test[test_idx].item()
ax[1][0].set_title(f"Test image ({label})", fontsize=8)
ax[1][0].axis("off")
for i in range(1, 7):
  ax[1][i].imshow(x_train[scores[-i][0]].reshape(28,28), cmap="gray", interpolation="nearest")
  label = y_train[scores[-i][0]]
  if_score = scores[-i][1].item()
  ax[1][i].set_title(f"Helpful image {scores[-i][0]} ({label})\nIF={round(if_score,3)}", fontsize=6)
  ax[1][i].axis("off")

plt.tight_layout()
plt.show()
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

import evaluate

from datautils import load_mnist

def load_data(batch_size: int, flatten: bool, pos_class: int, neg_class: int):
  train_loader, test_loader = load_mnist(batch_size=batch_size,
                                         flatten=flatten,
                                         subset_labels=(pos_class, neg_class))
  x_train, y_train = train_loader.dataset.tensors
  x_test, y_test = test_loader.dataset.tensors
  return (x_train, y_train), (x_test, y_test)

class BaseInfluenceModule:
  # for mnist example
  def __init__(self, model: nn.Module, pos_class: int, neg_class: int, damp: float = 1e-2):
    self.model = model
    self.pos_class = pos_class 
    self.neg_class = neg_class 
    self.damp = damp

  def compute_cost(self, xs, ys, element_wise=False):
    return self._compute_cost(xs, ys, theta_=self.model.fc.weight, element_wise=element_wise)

  def _compute_cost(self, xs, ys, theta_, element_wise=False):
    yhat = self.model(xs)
    ys = torch.where(ys == self.neg_class, 1., 0.).reshape(-1,1)
    if element_wise:
      cost = torch.nn.functional.binary_cross_entropy(yhat, ys, reduction="none")
      # cost = -torch.sum(ys*torch.log(yhat) + (1-ys)*torch.log(1-yhat), dim=1)
    else:
      cost = torch.nn.functional.binary_cross_entropy(yhat, ys, reduction="mean")
      # cost = -torch.sum(ys*torch.log(yhat) + (1-ys)*torch.log(1-yhat), dim=1).mean()
      if self.damp is not None:
        cost += self.damp * theta_.pow(2).sum().sqrt()

    return cost

  def evaluate_model(self, xs, ys, metric: str = "accuracy"):
    if metric == "accuracy":
      return self.compute_accuracy(xs, ys)
    else:
      raise ValueError(f"unsupported metric: {metric}")

  def compute_accuracy(self, xs, ys):
    acc = evaluate.load("accuracy")    
    yhat = self.model(xs).reshape(-1)
    preds = [1 if x > 0.5 else 0 for x in yhat]
    return acc.compute(references=ys, predictions=preds)["accuracy"]

  def _compute_grad(self, xs, ys):
    cost = self.compute_cost(xs, ys)
    grad = torch.autograd.grad(cost, self.model.fc.weight)
    if len(grad) == 1:
      grad = grad[0]
    else:
      raise ValueError(f"unknown grad shape: {grad.shape}")
    
    return cost, grad

  def _compute_hessian(self, xs, ys):
    def f(theta_): return self._compute_cost(xs, ys, theta_)
    hessian = torch.autograd.functional.hessian(f, self.model.fc.weight).squeeze()
    return hessian

  def _compute_inv_hessian(self, xs, ys):
    return torch.inverse(self._compute_hessian(xs, ys))




if __name__ == "__main__":

  pos_class = 0
  neg_class = 1
  damp = 1e-2

  # data
  (x_train, y_train), (x_test, y_test) = load_data(batch_size=1, flatten=True, pos_class=0, neg_class=1)

  # baseline model
  from sklearn.linear_model import LogisticRegression
  # TODO: C needed?
  model = LogisticRegression(fit_intercept=False, penalty="l2", C=damp) 
  # model = LogisticRegression(fit_intercept=False, penalty=None) 
  model = model.fit(x_train, y_train)

  fc = nn.Linear(28*28, 1, bias=False)
  fc.weight = nn.Parameter(torch.tensor(model.coef_, dtype=torch.float32))

  model = nn.Sequential()
  model.add_module("fc", fc)
  model.add_module("sigmoid", nn.Sigmoid())
  print(model)

  module = BaseInfluenceModule(model, pos_class=0, neg_class=1, damp=damp)
  test_acc = module.compute_acc(x_test, y_test)
  print(test_acc)

  inv_hessian = module._compute_inv_hessian(x_train, y_train)

  test_idx = [1778]
  test_cost, test_grad = module._compute_grad(x_test[test_idx], y_test[test_idx])
  ivhp = test_grad[0] @ inv_hessian # s_test
  
  scores = []
  for i in trange(len(x_train)):
    train_cost, train_grad = module._compute_grad(x_train[[i]], y_train[[i]])
    score = train_grad[0] @ ivhp.T
    score /= len(x_train)
    scores.append((i, score))

  scores = sorted(scores, key=lambda x: x[1]) # top scores = harmful, bottom scores = helpful

  scores_only = [s[1].item() for s in scores]

  scores_gaps = [scores_only[i]-scores_only[i-1] for i in range(1,len(scores_only))]
  _, (ax1, ax2) = plt.subplots(ncols=2)
  ax1.hist(scores_only, density=True)
  ax2.hist(scores_gaps, density=True)



  ## plotting
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

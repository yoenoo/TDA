import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import trange
from pathlib import Path
from typing import Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import SGD 
from torch.utils.data import DataLoader

from models import LRNet
from datautils import load_mnist
from influence import BaseInfluenceModule

__FIGS_PATH__ = "./figs"


## TODO: type annotation for module
## TODO: type annotation for optimizer
def train(module, train_loader: DataLoader, test_loader: Optional[DataLoader], 
          optimizer, epochs: int = 500):

  model = module.model
  model.train()
  for epoch in trange(epochs):
    for xs, ys in train_loader:
      optimizer.zero_grad()
      loss = module.compute_cost(xs, ys) 
      loss.backward()
      optimizer.step()

    ## TODO: put this inside pbar
    if (epoch % 100 == 0) and (epoch > 0):
      x_train, y_train = train_loader.dataset.tensors
      train_acc = module.evaluate_model(x_train, y_train, metric="accuracy")

      if test_loader is not None:
        x_test, y_test = test_loader.dataset.tensors
        test_acc = module.evaluate_model(x_test, y_test, metric="accuracy")
        test_loss = module.compute_cost(x_test, y_test)
        print(f"Train accuracy: {round(train_acc, 3)}, "
              f"Test accuracy: {round(test_acc, 3)}, "
              f"Test loss: {round(test_loss.item(), 3)}")
      else:
        print(f"Train accuracy: {round(train_acc, 3)}")

## TODO: type annotation
def influences(module, x_train, y_train, x_test, y_test, test_idx: int, sort=True):
  target_label = y_test[test_idx].item()

  inv_hessian = module._compute_inv_hessian(x_train, y_train)
  _, test_grad = module._compute_grad(x_test[[test_idx]], y_test[[test_idx]])
  ivhp = test_grad @ inv_hessian # s_test in Koh and Liang (2017)

  scores = []
  for i in trange(len(x_train)):
    _, train_grad = module._compute_grad(x_train[[i]], y_train[[i]])
    score = train_grad @ ivhp.T
    score /= len(x_train)
    scores.append((i, score.item()))

  if sort:
    scores = sorted(scores, key=lambda x: x[1]) 

  return scores

def find_chunk_stats(scores, test_idx, train_labels, test_labels, k=10, verbose=1):
  target_label = test_labels[test_idx].item()

  label_idxs = [s[0] for s in scores]  
  labels = [train_labels[i].item() for i in label_idxs]
  
  ## label distribution (harmful -> helpful)
  splits = np.array_split(labels, k)
  chunks = []
  for i, split in enumerate(splits):
    total_count = len(split)
    target_count = Counter(split)[target_label]
    target_prop = target_count / total_count
    chunks.append(target_prop)
    if verbose > 0:
      print(f"target proportions (chunk {i}): {target_prop:.3f}")

  return chunks

def plot_influence(scores, test_loss, normalize=True):
  scores_only = [s[1]/test_loss.detach().numpy() for s in scores] # normalize by test loss 
  scores_gaps = [scores_only[i]-scores_only[i-1] for i in range(1, len(scores_only))]
  _, (ax1, ax2) = plt.subplots(ncols=2)
  ax1.hist(scores_only, density=True); ax1.set_title("Influence scores")
  ax2.hist(scores_gaps, density=True); ax2.set_title("Influence score gaps")
  plt.tight_layout()

  Path(__FIGS_PATH__).mkdir(parents=True, exist_ok=True)
  # plt.savefig(f"{__FIGS_PATH__}/influence_score_distribution_{pos_class}_{neg_class}_{damp}_{epochs}.png")

def plot_examples(scores, test_idx, x_train, y_train, x_test, y_test):
  _, axes = plt.subplots(nrows=2, ncols=7, figsize=(16,4))
  config_kwargs = {"cmap": "gray", "interpolation": "nearest"}

  # top row = harmful
  ax1 = axes[0]
  ax1[0].imshow(x_test[test_idx].reshape(28,28), **config_kwargs)
  _label = y_test[test_idx].item()
  ax1[0].set_title(f"Test image ({_label})", fontsize=8)
  ax1[0].axis("off")
  for i in range(1,7):
    _label_idx, if_score = scores[i-1]
    _label = y_train[_label_idx]
    ax1[i].imshow(x_train[_label_idx].reshape(28,28), **config_kwargs)
    ax1[i].set_title(f"Harmful image {_label_idx} ({_label})\nIF={round(if_score,3)}", fontsize=8)
    ax1[i].axis("off")

  # bottom row = helpful
  ax2 = axes[1]
  ax2[0].imshow(x_test[test_idx].reshape(28,28), **config_kwargs)
  _label = y_test[test_idx].item()
  ax2[0].set_title(f"Test iamge ({_label})", fontsize=8)
  ax2[0].axis("off")
  for i in range(1,7):
    _label_idx, if_score = scores[-i]
    _label = y_train[_label_idx]
    ax2[i].imshow(x_train[_label_idx].reshape(28,28), **config_kwargs)
    ax2[i].set_title(f"Helpful image {_label_idx} ({_label})\nIF={round(if_score,3)}", fontsize=8)
    ax2[i].axis("off")

  plt.tight_layout() 
  # plt.savefig(f"{__FIGS_PATH__}/influential_examples_{pos_class}_{neg_class}_{damp}_{epochs}.png")


def run_lr_mlp_test(pos_class: int, neg_class: int, damp: float = 1e-2, epochs: int = 500):
  subset_labels = (pos_class, neg_class)
  train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)  
  (x_train, y_train), (x_test, y_test) = train_loader.dataset.tensors, test_loader.dataset.tensors

  model = LRNet()
  module = BaseInfluenceModule(model, pos_class, neg_class, damp)
  optimizer = SGD(model.parameters(), lr=1e-4)
  train(module, train_loader, test_loader, optimizer, epochs=epochs)

  model.eval()  

  ## validate test performance
  test_loss = module.compute_cost(x_test, y_test)

  test_idx = 1776 ## hand pick an example
  scores = influences(module, x_train, y_train, x_test, y_test, test_idx=test_idx)  

  ## label distribution (harmful -> helpful)
  chunk_stats = find_chunk_stats(scores, test_idx=test_idx, 
                                 train_labels=y_train, test_labels=y_test, 
                                 k=10, verbose=1)
  
  ## score and gap distribution
  plot_influence(scores, test_loss, normalize=True)
  
  ## helpful and harmful examples
  plot_examples(scores, test_idx, x_train, y_train, x_test, y_test)
  plt.show()

  return chunk_stats

def run_lr_scikit_learn_test(pos_class: int, neg_class: int, damp: float = 1e-2):
  subset_labels = (pos_class, neg_class)
  train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)  
  (x_train, y_train), (x_test, y_test) = train_loader.dataset.tensors, test_loader.dataset.tensors

  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression(fit_intercept=False, penalty="l2", C=damp) ## TODO: is right to use C?
  clf = clf.fit(x_train, y_train)

  model = LRNet()
  model.fc.weight = nn.Parameter(torch.tensor(clf.coef_, dtype=torch.float32))
  module = BaseInfluenceModule(model, pos_class, neg_class, damp)

  model.eval()  

  ## validate test performance
  test_loss = module.compute_cost(x_test, y_test)

  test_idx = 1776 ## hand pick an example
  scores = influences(module, x_train, y_train, x_test, y_test, test_idx=test_idx)  

  ## label distribution (harmful -> helpful)
  chunk_stats = find_chunk_stats(scores, test_idx=test_idx, 
                                 train_labels=y_train, test_labels=y_test, 
                                 k=10, verbose=1)
  
  ## score and gap distribution
  # plot_influence(scores, test_loss, normalize=True)
  
  ## helpful and harmful examples
  # plot_examples(scores, test_idx, x_train, y_train, x_test, y_test)

  return chunk_stats


def run():
  pass



if __name__ == "__main__":
  pos_class = 0
  neg_class = 1
  damp = 1e-2

  torch.manual_seed(123)  

  run_lr_mlp_test(pos_class, neg_class, damp, epochs=1000)

  # rounds = 10

  # random_weight_chunk_stats = []
  # epoch_1000_chunk_stats = []
  # scikit_learn_chunk_stats = []
  # for _ in range(rounds):
  #   tmp = run_lr_mlp_test(pos_class, neg_class, damp, epochs=0)
  #   random_weight_chunk_stats.append(tmp)

  #   tmp = run_lr_mlp_test(pos_class, neg_class, damp, epochs=1000)
  #   epoch_1000_chunk_stats.append(tmp)

  #   tmp = run_lr_scikit_learn_test(pos_class, neg_class, damp)
  #   scikit_learn_chunk_stats.append(tmp)

  # import pandas as pd 
  # columns = [f"chunk{i}" for i in range(10)]
  # df1 = pd.DataFrame(random_weight_chunk_stats, columns=columns)
  # df1["_label"] = "random_weights"
  # df2 = pd.DataFrame(epoch_1000_chunk_stats, columns=columns)
  # df2["_label"] = "epoch_1000"
  # df3 = pd.DataFrame(scikit_learn_chunk_stats, columns=columns)
  # df3["_label"] = "scikit_learn"
  # df = pd.concat([df1, df2, df3])
  # df.to_csv("chunk_stats.csv", index=False)
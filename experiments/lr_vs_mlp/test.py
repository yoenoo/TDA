import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import trange
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.optim import SGD 
from torch.utils.data import DataLoader

from models import LRNet
from datautils import load_mnist
from influence import BaseInfluenceModule

__FIGS_PATH__ = "./figs"

import argparse
from argparse import BooleanOptionalAction
parser = argparse.ArgumentParser()
parser.add_argument("--tqdm", default=True, action=BooleanOptionalAction)
parser.add_argument("--pos_class", type=int, required=True)
parser.add_argument("--neg_class", type=int, required=True)
FLAGS,_ = parser.parse_known_args()


## TODO: type annotation for module
## TODO: type annotation for optimizer
def train(module, train_loader: DataLoader, test_loader: Optional[DataLoader], 
          optimizer, epochs: int = 500):

  model = module.model
  model.train()
  for epoch in trange(epochs, disable=FLAGS.tqdm):
    for xs, ys in train_loader:
      optimizer.zero_grad()
      loss = module.compute_cost(xs, ys, train=True)
      loss.backward()
      optimizer.step()

    ## TODO: put this inside pbar
    if (epoch % 100 == 0) and (epoch > 0):
      x_train, y_train = train_loader.dataset.tensors
      train_acc = module.evaluate_model(x_train, y_train, metric="accuracy")

      if test_loader is not None:
        x_test, y_test = test_loader.dataset.tensors
        test_acc = module.evaluate_model(x_test, y_test, metric="accuracy")
        test_loss = module.compute_cost(x_test, y_test, train=False).item()
        print(f"Train accuracy: {round(train_acc, 3)}, "
              f"Test accuracy: {round(test_acc, 3)}, "
              f"Test loss: {round(test_loss, 3)}")
      else:
        print(f"Train accuracy: {round(train_acc, 3)}")

## TODO: type annotation
def influences(module, x_train, y_train, x_test, y_test, test_idx: int, sort=True):
  inv_hessian = module._compute_inv_hessian(x_train, y_train, train=True)
  _, test_grad = module._compute_grad(x_test[[test_idx]], y_test[[test_idx]], train=False)
  # ivhp = test_grad @ inv_hessian # s_test in Koh and Liang (2017)
  ivhp = inv_hessian @ test_grad.T # s_test in Koh and Liang (2017)

  scores = []
  for i in trange(len(x_train), disable=FLAGS.tqdm):
    _, train_grad = module._compute_grad(x_train[[i]], y_train[[i]], train=True)
    # score = train_grad @ ivhp.T
    score = train_grad @ ivhp
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

# def plot_influence(scores, test_loss, normalize=True):
def plot_influence(scores, save_as=None):
  # scores_only = [s[1]/test_loss.detach().numpy() for s in scores] # normalize by test loss 
  scores_only = [s[1] for s in scores]
  scores_gaps = [scores_only[i]-scores_only[i-1] for i in range(1, len(scores_only))]
  _, (ax1, ax2) = plt.subplots(ncols=2)
  ax1.hist(scores_only, density=True); ax1.set_title("Influence scores")
  ax2.hist(scores_gaps, density=True); ax2.set_title("Influence score gaps")
  plt.tight_layout()

  Path(__FIGS_PATH__).mkdir(parents=True, exist_ok=True)
  if save_as is not None:
    plt.savefig(f"{__FIGS_PATH__}/{save_as}")
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
  Path(__FIGS_PATH__).mkdir(parents=True, exist_ok=True)
  # plt.savefig(f"{__FIGS_PATH__}/influential_examples_{pos_class}_{neg_class}_{damp}_{epochs}.png")
  # test_idx = 1776 ## hand pick an example (previous example)


"""
def get_indices(test_loss, y_test, label):
  test_loss_sorted = torch.sort(test_loss)
  y_test_sorted = y_test[test_loss_sorted.indices]

  y_test_sorted_class_ind = torch.where(y_test_sorted == label)[0]
  best_idx  = y_test_sorted_class_ind[0].item()
  worst_idx = y_test_sorted_class_ind[-1].item()
  mid_idx   = y_test_sorted_class_ind[len(y_test_sorted_class_ind)//2].item()
  return best_idx, worst_idx, mid_idx
"""

def get_indices(test_loss, y_test, label):
  test_loss_class = torch.where(y_test==label, test_loss, np.nan)
  test_loss_class_sorted = torch.sort(test_loss_class)

  idx_cutoff = test_loss_class_sorted.values.isnan().nonzero()[0].item()
  idxs = test_loss_class_sorted.indices[:idx_cutoff]
  return idxs[0].item(), idxs[-1].item(), idxs[len(idxs)//2].item()


############################# Tests
def run_lr_mlp_test(pos_class: int, neg_class: int, target_class: int, damp: float = 1e-2, epochs: int = 500):
  subset_labels = (pos_class, neg_class)
  train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)  
  (x_train, y_train), (x_test, y_test) = train_loader.dataset.tensors, test_loader.dataset.tensors

  model = LRNet()
  module = BaseInfluenceModule(model, pos_class, neg_class, damp)
  optimizer = SGD(model.parameters(), lr=1e-4)
  train(module, train_loader, test_loader, optimizer, epochs=epochs)

  model.eval()  

  ## validate test performance
  test_loss = module.compute_cost(x_test, y_test, elementwise=True, train=False)
  best_example, worst_example, mid_example = get_indices(test_loss, y_test, target_class)  

  out = defaultdict(lambda: defaultdict(list))
  samples = {"worst": worst_example, "best": best_example, "median": mid_example}
  print(samples)
  for class_label, test_idx in samples.items():
    scores = influences(module, x_train, y_train, x_test, y_test, test_idx=test_idx)  
    quantiles = torch.quantile(torch.tensor(scores), q=torch.linspace(0,1,steps=11), dim=0)[:,1].tolist()
    out[class_label]["quantiles"] = quantiles

    ## label distribution (harmful -> helpful)
    chunk_stats = find_chunk_stats(scores, test_idx=test_idx, 
                                   train_labels=y_train, test_labels=y_test, 
                                   k=10, verbose=0)
    out[class_label]["chunk_stats"] = chunk_stats
    
    # ## score and gap distribution
    # plot_influence(scores, save_as=f"influence_functions_{class_label}_{test_idx}_epoch_{epochs}.png")
    
    # ## helpful and harmful examples
    # plot_examples(scores, test_idx, x_train, y_train, x_test, y_test)
    # plt.show()

  test_acc = module.evaluate_model(x_test, y_test, metric="accuracy")
  return out, test_acc

def run_lr_scikit_learn_test(pos_class: int, neg_class: int, target_class: int, damp: float = 1e-2):
  subset_labels = (pos_class, neg_class)
  train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)  
  (x_train, y_train), (x_test, y_test) = train_loader.dataset.tensors, test_loader.dataset.tensors

  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression(fit_intercept=False, penalty="l2", C=1e-3) ## TODO: is right to use C?
  # clf = LogisticRegression(fit_intercept=False, penalty=None)
  clf = clf.fit(x_train, y_train)

  model = LRNet()
  model.fc.weight = nn.Parameter(torch.tensor(clf.coef_, dtype=torch.float32))
  module = BaseInfluenceModule(model, pos_class, neg_class, damp)

  model.eval()  

  ## validate test performance
  test_loss = module.compute_cost(x_test, y_test, elementwise=True, train=False)
  best_example, worst_example, mid_example = get_indices(test_loss, y_test, target_class)

  out = defaultdict(lambda: defaultdict(list))
  samples = {"worst": worst_example, "best": best_example, "median": mid_example}
  print(samples)
  for class_label, test_idx in samples.items():
    scores = influences(module, x_train, y_train, x_test, y_test, test_idx=test_idx)  
    quantiles = torch.quantile(torch.tensor(scores), q=torch.linspace(0,1,steps=11), dim=0)[:,1].tolist()
    out[class_label]["quantiles"] = quantiles

    ## label distribution (harmful -> helpful)
    chunk_stats = find_chunk_stats(scores, test_idx=test_idx, 
                                   train_labels=y_train, test_labels=y_test, 
                                   k=10, verbose=0)
    out[class_label]["chunk_stats"] = chunk_stats

 
  ## score and gap distribution
  # plot_influence(scores, test_loss, normalize=True)
  
  ## helpful and harmful examples
  # plot_examples(scores, test_idx, x_train, y_train, x_test, y_test)

  train_acc = module.evaluate_model(x_train, y_train, metric="accuracy")
  test_acc = module.evaluate_model(x_test, y_test, metric="accuracy")
  return out, train_acc, test_acc


if __name__ == "__main__":
  pos_class = FLAGS.pos_class
  neg_class = FLAGS.neg_class

  # pos_class, neg_class = 0, 1
  # pos_class, neg_class = 0, 4
  # pos_class, neg_class = 0, 8
  damp = 1e-2

  seed = 123
  torch.manual_seed(seed)  

  target = pos_class

  # scikit-learn tests are fixed across different rounds
  scikit_learn_stats, train_acc, test_acc = run_lr_scikit_learn_test(pos_class=pos_class, 
                                                                     neg_class=neg_class, 
                                                                     damp=damp, 
                                                                     target_class=target)
  scikit_learn_quantiles = []
  scikit_learn_chunk_stats = []
  print(f"scikit-learn, train accuracy: {train_acc:.5f}, test accuracy: {test_acc:.5f}")
  for class_label, stats in scikit_learn_stats.items():
    scikit_learn_quantiles.append(["scikit-learn", class_label, 0, stats["quantiles"]])
    scikit_learn_chunk_stats.append(["scikit-learn", class_label, 0, stats["chunk_stats"]])
 
  rounds = 10
  # epoch_variations = [0, 1, 2, 5, 10, 100]
  epoch_variations = [0, 1, 3, 10, 100]
  mlp_quantiles = []
  mlp_chunk_stats = []
  for _round in range(rounds):
    for epoch in epoch_variations:
      tmp, test_acc = run_lr_mlp_test(pos_class=pos_class, 
                                      neg_class=neg_class, 
                                      damp=damp, 
                                      epochs=epoch, 
                                      target_class=target)
      print(f"epoch {epoch} (round: {_round}), test accuracy: {test_acc:.3f}")
      for class_label in ["best", "worst", "median"]:
        mlp_quantiles.append([f"mlp_{epoch}", class_label, _round, tmp[class_label]["quantiles"]])
        mlp_chunk_stats.append([f"mlp_{epoch}", class_label, _round, tmp[class_label]["chunk_stats"]])

  # aggregate stats
  chunk_stats = scikit_learn_chunk_stats + mlp_chunk_stats
  chunk_stats = pd.concat([
    pd.DataFrame(chunk_stats).iloc[:,:3], 
    pd.DataFrame(chunk_stats)[3].apply(pd.Series),
  ], axis=1)
  chunk_stats.columns = ["epoch", "class_label", "round"] + [f"chunk_{i}" for i in range(10)]
  chunk_stats.to_csv(f"chunk_stats_{pos_class}_{neg_class}_{seed}__092324.csv", index=False)

  quantile_stats = scikit_learn_quantiles + mlp_quantiles
  quantile_stats = pd.concat([
    pd.DataFrame(quantile_stats).iloc[:,:3],
    pd.DataFrame(quantile_stats)[3].apply(pd.Series),
  ], axis=1)
  quantile_stats.columns = ["epoch", "class_label", "round"] + [f"quantile_{q}" for q in range(11)]
  quantile_stats.to_csv(f"quantile_stats_{pos_class}_{neg_class}_{seed}__092324.csv", index=False)
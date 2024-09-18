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




def run_lr_mlp_test(pos_class: int, neg_class: int, damp: float = 1e-2, epochs: int = 500):
  subset_labels = (pos_class, neg_class)
  train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)  
  (x_train, y_train), (x_test, y_test) = train_loader.dataset.tensors, test_loader.dataset.tensors

  model_skeleton = LRNet()

  from sklearn.linear_model import LogisticRegression  
  clf = LogisticRegression(fit_intercept=False, penalty="l2", C=damp) ## TODO: is right to use C?
  clf = clf.fit(x_train, y_train)

  # ground truth
  baseline_model = model_skeleton
  baseline_model.fc.weight = nn.Parameter(torch.tensor(clf.coef_, dtype=torch.float32))

  # MLP
  model = 

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
  # plot_influence(scores, test_loss, normalize=True)
  
  ## helpful and harmful examples
  # plot_examples(scores, test_idx, x_train, y_train, x_test, y_test)

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

  rounds = 10

  random_weight_chunk_stats = []
  epoch_1000_chunk_stats = []
  scikit_learn_chunk_stats = []
  for _ in range(rounds):
    tmp = run_lr_mlp_test(pos_class, neg_class, damp, epochs=0)
    random_weight_chunk_stats.append(tmp)

    tmp = run_lr_mlp_test(pos_class, neg_class, damp, epochs=1000)
    epoch_1000_chunk_stats.append(tmp)

    tmp = run_lr_scikit_learn_test(pos_class, neg_class, damp)
    scikit_learn_chunk_stats.append(tmp)

  import pandas as pd 
  columns = [f"chunk{i}" for i in range(10)]
  df1 = pd.DataFrame(random_weight_chunk_stats, columns=columns)
  df1["_label"] = "random_weights"
  df2 = pd.DataFrame(epoch_1000_chunk_stats, columns=columns)
  df2["_label"] = "epoch_1000"
  df3 = pd.DataFrame(scikit_learn_chunk_stats, columns=columns)
  df3["_label"] = "scikit_learn"
  df = pd.concat([df1, df2, df3])
  df.to_csv("chunk_stats.csv", index=False)
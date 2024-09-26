import torch
import numpy as np 
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from datautils import load_mnist


def load_data(pos_class, neg_class):
  subset_labels = (pos_class, neg_class)
  train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)  
  return train_loader.dataset.tensors, test_loader.dataset.tensors

def create_model(**kwargs):
  # C = 1.0 / (self.num_train_examples * self.weight_decay)
  return LogisticRegression(
      # C=C,
      tol=1e-8,
      fit_intercept=False,
      solver="lbfgs",
      warm_start=True,
      max_iter=1000,
      **kwargs,
  )
  
def train_model(model, x_train, y_train, x_test, y_test):
  model.fit(x_train, y_train)

  y_pred = model.predict(x_train)
  train_acc = accuracy_score(y_train, y_pred) 

  y_pred = model.predict(x_test)
  test_acc = accuracy_score(y_test, y_pred)
  print(f"train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

def get_chunk_stats(y_train, indices, target_label, k=10):
  chunks = np.array_split(indices, k)
  props = []
  for chunk in chunks:
      labels = y_train[chunk].tolist()
      vals = Counter(labels)[target_label] / len(labels)
      props.append(vals)

  return props

def sigmoid(x): return 1 / (1 + torch.exp(-x))

if __name__ == "__main__":
  pos_class = 2
  neg_class = 3

  (x_train, y_train), (x_test, y_test) = load_data(pos_class, neg_class)

  # label transformation
  y_train = torch.tensor([1 if y == neg_class else 0 for y in y_train])
  y_test = torch.tensor([1 if y == neg_class else 0 for y in y_test])

  model = create_model()
  train_model(model, x_train, y_train, x_test, y_test) 
  W = torch.tensor(model.coef_.T, dtype=torch.float)
  z = x_train @ W
  h = sigmoid(z)

  d_ii = (h*(1-h)).squeeze()
  D = torch.diag(d_ii)

  yhat = torch.tensor(model.predict_proba(x_test)[:,0]).to(torch.float)
  # yhat = torch.tensor(model.predict(x_test)).to(float)
  # yhat = torch.tensor([1. if y == neg_class else 0. for y in yhat])
  
  test_loss = torch.nn.functional.binary_cross_entropy(yhat, y_test.to(torch.float), reduction="none")
  best_idx, worst_idx = test_loss.argmin().item(), test_loss.argmax().item()
  print(best_idx, worst_idx)

  damps = 10 ** np.linspace(-5,2,8)
  
  # for visualization
  examples = defaultdict(list)

  # stats = defaultdict(list)
  for damp in damps:
    print(f"damp: {damp}")
    hess = (x_train.T @ D @ x_train) / len(x_train) + damp * torch.eye(len(W))

    eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
    print(f"hessian min eigval {np.min(eigvals).item()}")
    print(f"hessian max eigval {np.max(eigvals).item()}")

    for test_idx in [best_idx, worst_idx]:
      xt = x_test[test_idx]
      yt = y_test[test_idx] * 2 - 1
      inv_hess = torch.inverse(hess)

      influences = []
      for i in range(len(x_train)):
        xtt = x_train[i]
        ytt = y_train[i] * 2 - 1
        score = yt * ytt * sigmoid(-yt * xt @ W) * sigmoid(-ytt * xt @ W) * xtt @ inv_hess @ xtt.T
        influences.append((i,score))

      influences = torch.tensor(influences)
      values, indices = torch.sort(influences[:,1])

      examples[damp].append([test_idx, indices[:5].tolist()]) # harmful
      examples[damp].append([test_idx, indices[-5:].tolist()]) # helpful

      # # quantiles
      # quantiles = torch.quantile(torch.tensor(values), q=torch.linspace(0,1,steps=101)).tolist()
      # stats[damp].append([test_idx, quantiles])

      # # chunk stats
      # target_label = y_test[test_idx].item()
      # cstats = get_chunk_stats(y_train, indices, target_label) 
      # stats[damp].append([test_idx, cstats])
 
  # print(dict(stats))



# TODO: show image samples
print(dict(examples))
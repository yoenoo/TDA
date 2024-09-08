from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

def _set_attr(obj, names, val):
  if len(names) == 1:
    setattr(obj, names[0], val)
  else:
    _set_attr(getattr(obj, names[0]), names[1:], val)

def _del_attr(obj, names):
  if len(names) == 1:
    delattr(obj, names[0])
  else:
    _del_attr(getattr(obj, names[0]), names[1:])

class BaseInfluenceModule(ABC):
  def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
    self.model = model

    self.is_model_functional = False
    #self.params_names  = tuple(name for name, _ in self._model_params())
    #self.params_shapes = tuple(p.shape for _, p in self._model_params())
    self.params_names  = self._param_names()
    self.params_shapes = self._param_shapes()

    self.train_loader = train_loader
    self.test_loader  = test_loader

  @abstractmethod
  def inverse_hvp(self, vec: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError()

  def stest(self, test_idxs: list[int]) -> torch.Tensor:
    return self.inverse_hvp(self.test_loss_grad(test_idxs))

  def train_loss_grad(self, train_idxs: list[int]) -> torch.Tensor:
    return self._loss_grad(train_idxs, is_train=True)
  
  def test_loss_grad(self, test_idxs: list[int]) -> torch.Tensor:
    return self._loss_grad(test_idxs, is_train=False)

  def influences(self, train_idxs: list[int], test_idxs: list[int]) -> torch.Tensor:
    stest = self.stest(test_idxs)
    scores = []
    for grad_z, _ in self._loss_grad_loader(batch_size=1, subset=train_idxs, is_train=True):
      s = grad_z @ stest
      scores.append(s)
    return torch.tensor(scores) / len(self.train_loader.dataset)


  # -- helper functions
  def _model_params(self, with_names=True):
    assert not self.is_model_functional
    return tuple((name, p) if with_names else p for name, p in self.model.named_parameters() if p.requires_grad) 

  def _param_names(self):
    return tuple(name for name, _ in self._model_params())
  
  def _param_shapes(self):
    return tuple(p.shape for _, p in self._model_params())

  def _make_functional(self):
    assert not self.is_model_functional
    params = tuple(p.detach().requires_grad_() for p in self._model_params(False))

    for name in self.params_names:
      _del_attr(self.model, name.split("."))
    
    self.is_model_functional = True
    return params

  def _flatten_params_like(self, params_like):
    vec = []
    for p in params_like:
      vec.append(p.view(-1))
    return torch.cat(vec)

  def _reshape_like_params(self, vec):
    ptr = 0
    split_tensors = []
    for dim in self.params_shapes:
      num_param = dim.numel()
      split_tensors.append(vec[ptr:ptr+num_param].view(dim))
      ptr += num_param
    return tuple(split_tensors)

  def _model_reinsert_params(self, params, register=False):
    for name, p in zip(self.params_names, params):
      _set_attr(self.model, name.split("."), torch.nn.Parameter(p) if register else p)
    
    self.is_model_functional = not register


  def _loader(self, is_train, batch_size=None, subset=None):
    loader = self.train_loader if is_train else self.test_loader
    batch_size = loader.batch_size if batch_size is None else batch_size

    if subset is None:
      dataset = loader.dataset
    else:
      dataset = Subset(loader.dataset, indices=subset)

    new_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    yield from new_loader

  def _loss_grad_loader(self, is_train, batch_size=None, subset=None):
    params = self._model_params(False)
    flat_params = self._flatten_params_like(params)

    # TODO: make it more general with e.g. loss_fn
    for data, target in self._loader(subset=subset, batch_size=batch_size, is_train=is_train):
      _target = torch.where(target == 7, 1., 0.).reshape(-1,1)
      _output = self.model(data.reshape(-1, 28*28))
      loss = F.binary_cross_entropy(_output, _target)
      batch_size = len(data)
      yield self._flatten_params_like(torch.autograd.grad(loss, params)), batch_size

  def _loss_grad(self, idxs, is_train):
    grad = 0
    for batch_grad, batch_size in self._loss_grad_loader(subset=idxs, is_train=is_train):
      grad += batch_grad * batch_size
    return grad / len(idxs)
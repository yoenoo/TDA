from .base import BaseInfluenceModule

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


CHECK_EIGVALS = True

class AutogradInfluenceModule(BaseInfluenceModule):
  def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, damp: float):
    super().__init__(model, train_loader, test_loader)
    self.damp = damp
    
  def inverse_hvp(self, vec):
    params = self._make_functional()
    flat_params = self._flatten_params_like(params)
    d = len(flat_params)

    hess = 0
    for data, target in self._loader(is_train=True):
      def f(theta_):
        self._model_reinsert_params(self._reshape_like_params(theta_))
        _output = self.model(data.reshape(-1, 28*28))
        _target = torch.where(target==7, 1., 0,).reshape(-1,1)
        return F.binary_cross_entropy(_output, _target)

      hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
      hess += hess_batch * len(data) # TODO: why multiply by batch_size?

    with torch.no_grad():
      self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
      hess /= len(self.train_loader)
      hess += self.damp * torch.eye(d) # TODO: why need to do this?

      if CHECK_EIGVALS:
        eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
        print(f"hessian min eigval {np.min(eigvals).item()}")
        print(f"hessian max eigval {np.max(eigvals).item()}")
        if not np.all(eigvals >= 0):
          raise ValueError()
      
      inverse_hess = torch.inverse(hess)
      
    return inverse_hess @ vec

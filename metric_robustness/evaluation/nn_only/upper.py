import torch
from torch import Tensor


class UpperSolver:

    def __init__(
            self, X_train: Tensor, y_train: Tensor
    ):
        self._X_train = X_train
        self._y_train = y_train

    def __call__(self, x_eval: Tensor, y_eval: Tensor):
        mask = y_eval != self._y_train
        return torch.norm(x_eval - self._X_train[mask], dim=1).min().item()

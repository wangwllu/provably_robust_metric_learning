"""
scipy.spatial.distance.cdist has very poor performance for mahalanobis,
so does KNeighborsClassifier.
So, I have to implement 1-NN from scratch
"""


import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional
from abc import ABC, abstractmethod


def cross_squared_euclidean_distances(X: Tensor, Y: Tensor):
    X_product = (X ** 2).sum(dim=1)
    Y_product = (Y ** 2).sum(dim=1)
    cross_product = X @ Y.t()
    return (
        X_product.unsqueeze(1)
        + Y_product.unsqueeze(0)
        - 2 * cross_product
    )


def cross_squared_mahalanobis_distances(
        X: Tensor, Y: Tensor, M: Tensor,
        cache: Optional[Tensor] = None
):

    if cache is None:
        X_product = ((X @ M) * X).sum(dim=1)
    else:
        X_product = (cache * X).sum(dim=1)

    Y_product = ((Y @ M) * Y).sum(dim=1)

    cross_product = X @ M @ Y.t()
    return (
        X_product.unsqueeze(1)
        + Y_product.unsqueeze(0)
        - 2 * cross_product
    )


class Predictor(ABC):

    @abstractmethod
    def predict_batch(self, X_eval: Tensor):
        pass

    def predict_individual(self, x_eval: Tensor):
        return self.predict_batch(x_eval.unsqueeze(0)).item()

    def score(self, X_eval: Tensor, y_eval: Tensor):

        BATCH_SIZE = 1000
        y_pred = torch.cat(
            [self.predict_batch(X_eval_batch)
             for X_eval_batch in torch.split(X_eval, BATCH_SIZE)]
        )
        return (y_pred == y_eval).sum().item() / y_eval.shape[0]

        # return (
        #     self.predict_batch(X_eval) == y_eval
        # ).sum().item() / y_eval.shape[0]

    def _score_helper(self, X_eval: Tensor):
        return self.predict_batch(X_eval)


class EuclideanNearestNeighborPredictor(Predictor):

    def __init__(
            self, X_train: Tensor, y_train: Tensor,
    ):
        self._X_train = X_train
        self._y_train = y_train

    def predict_batch(self, X_eval: Tensor):
        csd = self.cross_squared_distances(X_eval)
        indices = torch.argmin(csd, dim=0)
        return torch.take(self._y_train, indices)

    def cross_squared_distances(self, X_eval: Tensor):
        return cross_squared_euclidean_distances(
            self._X_train, X_eval
        )

    def on_boundary(self, x_eval, y_eval):
        distances = self.cross_squared_distances(
            x_eval.unsqueeze(0)
        ).squeeze()

        mask = y_eval == self._y_train
        p_distances = distances[mask]
        n_distances = distances[~mask]

        # print(p_distances.min(), n_distances.min())

        return torch.isclose(
            p_distances.min(), n_distances.min(), rtol=1e-3, atol=1e-6
        )


class EuclideanKnnPredictor(EuclideanNearestNeighborPredictor):

    def __init__(self, X_train: Tensor, y_train: Tensor, n_neighbors: int):
        super().__init__(X_train, y_train)
        self._n_neighbors = n_neighbors

    def predict_batch(self, X_eval: Tensor):
        csd = self.cross_squared_distances(X_eval)
        _, indices = csd.t().topk(
            self._n_neighbors, dim=1, largest=False, sorted=False
        )
        return torch.take(self._y_train, indices).mode(dim=1)[0]

    def on_boundary(self, x_eval, y_eval):
        raise Exception('Unimplemneted!')


class MahalanobisNearestNeighborPredictor(EuclideanNearestNeighborPredictor):

    def __init__(
        self, X_train, y_train, M
    ):
        super().__init__(X_train, y_train)
        self._M = M
        self._cache = self._X_train @ self._M

    def cross_squared_distances(self, X_eval):
        return cross_squared_mahalanobis_distances(
            self._X_train, X_eval, self._M, self._cache
        )


class MahalanobisKnnPredictor(EuclideanKnnPredictor):
    def __init__(
        self, X_train, y_train, n_neighbors, M
    ):
        super().__init__(X_train, y_train, n_neighbors)
        self._M = M
        self._cache = self._X_train @ self._M

    def cross_squared_distances(self, X_eval):
        return cross_squared_mahalanobis_distances(
            self._X_train, X_eval, self._M, self._cache
        )


class NearestNeighborPredictorFactory:

    def create(self, metric, X_train, y_train, M=None):
        if metric == 'euclidean':
            return EuclideanNearestNeighborPredictor(
                X_train, y_train
            )
        elif metric == 'mahalanobis':
            assert M is not None
            return MahalanobisNearestNeighborPredictor(
                X_train, y_train, M
            )
        else:
            raise Exception('unsupported nearest neighbor predictor')


class KnnFactory:

    def create(self, metric, X_train, y_train, n_neighbors=1, M=None):
        if metric == 'euclidean':
            return EuclideanKnnPredictor(
                X_train, y_train, n_neighbors
            )
        elif metric == 'mahalanobis':
            assert M is not None
            return MahalanobisKnnPredictor(
                X_train, y_train, n_neighbors, M
            )
        else:
            raise Exception('unsupported nearest neighbor predictor')


class MahalanobisKnnModule(nn.Module):

    def __init__(
        self, X_train: Tensor, y_train: Tensor, n_neighbors: int, M: Tensor
    ):
        super().__init__()

        self._X_train = nn.Parameter(X_train, requires_grad=False)
        self._y_train = nn.Parameter(y_train, requires_grad=False)
        self._M = nn.Parameter(M, requires_grad=False)
        self._cache = nn.Parameter(X_train @ M, requires_grad=False)

        self._n_neighbors = n_neighbors
        # self._n_labels = int(y_train.max().item()) + 1
        self._n_labels = int(y_train.max().item() + 1)

        self._uniques = nn.Parameter(
            torch.arange(
                self._n_labels, device=X_train.device
            ), requires_grad=False
        )

        self._eps = 1 / self._n_labels

    def forward(self, X_eval):
        csd = cross_squared_mahalanobis_distances(
            self._X_train, X_eval, self._M, self._cache)
        _, indices = csd.t().topk(
            self._n_neighbors, dim=1, largest=False, sorted=False
        )

        neighbor_lables = torch.take(self._y_train, indices)
        # unique_labels = torch.arange(
        #     self._n_labels, device=X_eval.device
        # ).unsqueeze(-1).unsqueeze(-1)

        # the last line is really tricky
        # argmax is terrible because it prefers the latter index which is different from mode
        # assert (self._n_labels-1) * 1e-5 < 1
        # eps = 1 / self._n_labels
        return (
            (neighbor_lables == self._uniques.unsqueeze(-1).unsqueeze(-1)).sum(dim=-1).t().to(torch.float)
            - self._uniques.to(torch.float).unsqueeze(0) * self._eps
        )

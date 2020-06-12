"""
scipy.spatial.distance.cdist has very poor performance for mahalanobis,
so does KNeighborsClassifier.
So, I have to implement 1-NN from scratch
"""


import numpy as np
from abc import ABC, abstractmethod


def compute_cross_squared_euclidean_distances(X, Y):
    X_product = (X ** 2).sum(axis=1)
    Y_product = (Y ** 2).sum(axis=1)
    cross_product = X @ Y.T
    return (
        X_product[:, np.newaxis]
        + Y_product[np.newaxis, :]
        - 2 * cross_product
    )


def compute_cross_squared_mahalanobis_distances(X, Y, M, cache=None):

    if cache is None:
        X_product = ((X @ M) * X).sum(axis=1)
    else:
        X_product = (cache * X).sum(axis=1)

    Y_product = ((Y @ M) * Y).sum(axis=1)

    cross_product = X @ M @ Y.T
    return (
        X_product[:, np.newaxis]
        + Y_product[np.newaxis, :]
        - 2 * cross_product
    )


class Predictor(ABC):

    @abstractmethod
    def predict_batch(self, X_eval):
        pass

    def predict_individual(self, x_eval):
        return self.predict_batch(x_eval[np.newaxis, :]).item()

    def score(self, X_eval, y_eval):
        return (self.predict_batch(X_eval) == y_eval).sum() / y_eval.shape[0]


class EuclideanNearestNeighborPredictor(Predictor):

    def __init__(
            self, X_train, y_train,
    ):
        self._X_train = X_train
        self._y_train = y_train

    def predict_batch(self, X_eval):
        csd = self.compute_cross_squared_distances(X_eval)
        indices = np.argmin(csd, axis=0)
        return np.take(self._y_train, indices)

    def compute_cross_squared_distances(self, X_eval):
        return compute_cross_squared_euclidean_distances(
            self._X_train, X_eval
        )


class MahalanobisNearestNeighborPredictor(EuclideanNearestNeighborPredictor):

    def __init__(
        self, X_train, y_train, M
    ):
        super().__init__(X_train, y_train)
        self._M = M
        self._cache = self._X_train @ self._M

    def compute_cross_squared_distances(self, X_eval):
        return compute_cross_squared_mahalanobis_distances(
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

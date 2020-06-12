import numpy as np
from .predictors import Predictor


class BinarySearch:

    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def __call__(self, x_pos, y_pos, x_neg):
        while True:
            x_mid = (x_pos + x_neg) / 2
            if self._predictor.predict_individual(x_mid) == y_pos:
                x_pos = x_mid
            else:
                x_neg = x_mid
            if np.allclose(x_pos, x_neg):
                return x_neg

from abc import ABC, abstractmethod
import numpy as np


class QpSolver(ABC):

    @abstractmethod
    def __call__(Q, b):
        """ Solve min_a  0.5*aQa + b^T a s.t. a>=0
        """
        pass


class GcdQpSolver(QpSolver):

    def __init__(self, max_iter=20000, eps=1e-12):
        self._max_iter = max_iter
        self._eps = eps

    def __call__(self, Q, b):
        K = Q.shape[0]
        alpha = np.zeros((K,), dtype=Q.dtype)
        g = b
        Qdiag = np.diag(Q)
        for i in range(self._max_iter):
            delta = np.maximum(alpha - g/Qdiag.clip(min=self._eps), 0) - alpha
            idx = np.argmax(np.abs(delta))
            val = delta[idx]
            if abs(val) < self._eps:
                break
            g = g + val*Q[:, idx]
            alpha[idx] += val

        return alpha


class QpSolverFactory:

    def create(self, name):
        if name == 'gcd':
            return GcdQpSolver()
        else:
            raise Exception('unsupported qp solver')

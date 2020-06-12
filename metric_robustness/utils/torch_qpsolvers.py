from abc import ABC, abstractmethod
from torch import Tensor


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

    def __call__(self, Q: Tensor, b: Tensor):
        K = Q.shape[0]
        alpha = Q.new_zeros((K,))

        g = b
        Qdiag = Q.diag().clamp(min=self._eps)
        for i in range(self._max_iter):

            delta = (
                alpha - g / Qdiag
            ).clamp(min=0) - alpha

            idx = delta.abs().argmax()
            val = delta[idx]
            if val.abs().item() < self._eps:
                break
            g = g + val*Q[:, idx]
            alpha[idx] += val

        # test
        # print(f'n_iter: {i}')
        return alpha


class QpSolverFactory:

    def create(self, name):
        if name == 'gcd':
            return GcdQpSolver()
        else:
            raise Exception('unsupported qp solver')

import math
import numpy as np

from metric_robustness.utils.sorting import top_k_min_indices
from metric_robustness.utils.predictors import NearestNeighborPredictorFactory
from metric_robustness.utils.qpsolvers import QpSolverFactory


class ExactEuclideanSolver:

    def __init__(
            self, X_train, y_train, qp_solver=QpSolverFactory().create('gcd'),
            screening_check_size=None, bounded=False, upper=1., lower=0.,
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._qp_solver = qp_solver

        self._screening_check_size = screening_check_size
        self._bounded = bounded
        self._upper = upper
        self._lower = lower

        self._initialize_predictor()

    def _initialize_predictor(self):
        self._predictor = NearestNeighborPredictorFactory().create(
            'euclidean',
            self._X_train, self._y_train
        )

    def __call__(self, x_eval, y_eval: int):

        if self.predict_individual(x_eval) != y_eval:
            return np.zeros_like(x_eval)

        else:
            return self._compute_best_perturbation_for_correctly_classified(
                x_eval, y_eval
            )

    def predict_individual(self, x_eval):
        return self._predictor.predict_individual(x_eval)

    def _compute_best_perturbation_for_correctly_classified(
            self, x_eval, y_eval
    ):
        X_pos, X_neg = self._partition_pos_neg(x_eval, y_eval)
        X_screen = self._compute_pos_for_screen(x_eval, X_pos)

        cache = self._compute_cache(X_pos)

        best_perturbation = None
        min_perturbation_norm = math.inf

        # n_screens = 0

        for x_neg in self._sorted_neg_generator(x_eval, X_neg):
            if self._screenable(
                    x_eval, x_neg, X_screen, min_perturbation_norm
            ):
                # n_screens += 1
                continue
            else:
                perturbation = self._solve_subproblem(
                    x_eval, x_neg, X_pos, cache
                )
                perturbation_norm = np.linalg.norm(perturbation)
                if perturbation_norm < min_perturbation_norm:
                    min_perturbation_norm = perturbation_norm
                    best_perturbation = perturbation

        # print(f'# screens: {n_screens}')
        return best_perturbation

    def _partition_pos_neg(self, x_eval, y_eval):
        mask = (self._y_train == y_eval)
        X_pos = self._X_train[mask]
        X_neg = self._X_train[~mask]
        return X_pos, X_neg

    def _compute_pos_for_screen(self, x_eval, X_pos):
        if self._screening_check_size is None:
            return X_pos
        else:
            return self._compute_nearest_pos(x_eval, X_pos)

    def _compute_cache(self, X_pos):
        return X_pos @ X_pos.T

    def _compute_nearest_pos(self, x_eval, X_pos):

        assert self._screening_check_size <= X_pos.shape[0]

        indices = top_k_min_indices(
            np.linalg.norm(x_eval - X_pos, axis=1),
            self._screening_check_size
        )
        return X_pos[indices]

    def _sorted_neg_generator(self, x_eval, X_neg):
        indices = np.argsort(
            np.linalg.norm(
                X_neg - x_eval, axis=1
            )
        )
        for i in indices:
            yield X_neg[i]

    def _screenable(self, x_eval, x_neg, X_screen, threshold):
        return threshold <= np.max(
            np.maximum(
                np.sum(
                    np.multiply(
                        2 * x_eval - X_screen - x_neg, X_screen - x_neg
                    ),
                    axis=1
                ),
                0
            ) / (2 * np.linalg.norm(X_screen - x_neg, axis=1))
        )

    def _solve_subproblem(self, x_eval, x_neg, X_pos, cache):

        A, b, Q = self._compute_qp_params(
            x_eval, x_neg, X_pos, cache
        )

        # min 0.5 * v.T @ Q @ v + v.T @ b, v >= 0
        # max - 0.5 * v.T @ Q @ v - v.T @ b, v >= 0
        lamda = self._qp_solver(Q, b)
        return -A.T @ lamda

    def _compute_qp_params(
            self, x_eval, x_neg, X_pos, cache
    ):

        A, b, Q = self._compute_unbounded_qp_params(
            x_eval, x_neg, X_pos, cache
        )

        if not self._bounded:
            return A, b, Q

        else:
            return self._compute_bounded_qp_params(x_eval, X_pos, A, b, Q)

    def _compute_unbounded_qp_params(
            self, x_eval, x_neg, X_pos, cache
    ):
        # A @ u <= b
        A = 2 * (X_pos - x_neg)

        # test: this one is much more efficient due to less multiplications
        # b = np.sum(np.multiply(X_pos + x_neg - 2 * x_eval,
        #                        X_pos - x_neg), axis=1)

        b = (
            np.diag(cache)
            - 2 * X_pos @ x_eval
            + x_neg.T @ (2 * x_eval - x_neg)
        )

        # X @ y
        temp = X_pos @ x_neg

        # A @ A.T = 4 * (X @ X.T - X @ y - (X @ y).T + y.T @ y)
        Q = 4 * (cache - temp[np.newaxis, :]
                 - temp[:, np.newaxis] + x_neg @ x_neg)

        return A, b, Q

    def _compute_bounded_qp_params(self, x_eval, X_pos, A, b, Q):
        # upper bound
        # A1 @ delta <= b1
        # z + delta <= upper
        A1 = np.identity(X_pos.shape[1], dtype=X_pos.dtype)
        b1 = self._upper - x_eval

        # lower bound
        # A2 @ delta <= b2
        # z + delta >= lower
        A2 = -np.identity(X_pos.shape[1], dtype=X_pos.dtype)
        b2 = x_eval - self._lower

        # A_full @ A_full.T
        Q_full = np.block([
            [Q, A, -A],
            [A.T, A1, A2],
            [-A.T, A2, A1],
        ])

        A_full = np.block([
            [A],
            [A1],
            [A2]
        ])

        b_full = np.concatenate([b, b1, b2])
        return A_full, b_full, Q_full


class ExactMahalanobisSolver(ExactEuclideanSolver):

    def __init__(
            self, X_train, y_train, psd_matrix,
            qp_solver=QpSolverFactory().create('gcd'),
            screening_check_size=None, bounded=False, upper=1., lower=0.,
    ):
        self._psd_matrix = psd_matrix
        super().__init__(
            X_train, y_train,
            qp_solver, screening_check_size, bounded, upper, lower
        )

    def _initialize_predictor(self):
        self._predictor = NearestNeighborPredictorFactory().create(
            'mahalanobis', self._X_train, self._y_train,
            M=self._psd_matrix
        )

    def _compute_cache(self, X_pos):
        return (
            X_pos @ self._psd_matrix @ self._psd_matrix.T @ X_pos.T,
            ((X_pos @ self._psd_matrix) * X_pos).sum(axis=1)
        )

    def _compute_unbounded_qp_params(
            self, x_eval, x_neg, X_pos, cache
    ):

        A = 2 * (X_pos - x_neg) @ self._psd_matrix

        b = (
            cache[1]
            - 2 * X_pos @ self._psd_matrix @ x_eval
            + x_neg.T @ self._psd_matrix @ (2 * x_eval - x_neg)
        )

        # test
        # b_prime = np.array(
        #     [
        #         (x_pos - x_eval) @ self._psd_matrix @ (x_pos - x_eval)
        #         - (x_neg - x_eval) @ self._psd_matrix @ (x_neg - x_eval)
        #         for x_pos in X_pos
        #     ]
        # )
        # assert np.allclose(b, b_prime)

        temp = X_pos @ self._psd_matrix @ self._psd_matrix.T @ x_neg
        Q = 4 * (cache[0] - temp[np.newaxis, :]
                 - temp[:, np.newaxis]
                 + x_neg.T @ self._psd_matrix @ self._psd_matrix.T @ x_neg)

        # test
        # assert np.allclose(Q, A @ A.T)

        return A, b, Q


class ExactSolverFactory:

    def create(
            self, name, X_train, y_train, psd_matrix=None,
            qp_solver=QpSolverFactory().create('gcd'),
            screening_check_size=None, bounded=False, upper=1., lower=0.
    ):
        if name == 'euclidean':
            result = ExactEuclideanSolver(
                X_train, y_train,
                qp_solver=qp_solver,
                screening_check_size=screening_check_size,
                bounded=bounded, upper=upper, lower=lower
            )
        elif name == 'mahalanobis':
            assert psd_matrix is not None
            result = ExactMahalanobisSolver(
                X_train, y_train, psd_matrix,
                qp_solver=qp_solver,
                screening_check_size=screening_check_size,
                bounded=bounded, upper=upper, lower=lower
            )
        else:
            raise Exception('unsupported solver')

        return result

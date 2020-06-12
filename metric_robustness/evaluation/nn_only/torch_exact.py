import math
import torch
import numpy as np

from torch import Tensor


from metric_robustness.utils.torch_predictors \
    import NearestNeighborPredictorFactory
from metric_robustness.utils.torch_qpsolvers \
    import QpSolverFactory


class ExactEuclideanSolver:

    def __init__(
            self, X_train: np.array, y_train: np.array,
            qp_solver=QpSolverFactory().create('gcd'),
            screening_check_size=None, bounded=False, upper=1., lower=0.,
            device='cpu', verbose=True
    ):
        self._verbose = verbose
        self._initialize_device(device)

        self._X_train = self._new_tensor(X_train)
        self._y_train = self._new_tensor(y_train)

        self._qp_solver = qp_solver
        self._screening_check_size = screening_check_size
        self._bounded = bounded
        self._upper = upper
        self._lower = lower

        self._initialize_predictor()

    def _initialize_device(self, device):
        if not torch.cuda.is_available():
            self._device = 'cpu'
        else:
            self._device = device

        if self._verbose:
            print(f'The exact solver is using {self._device}!')

    def _new_tensor(self, X: np.array):
        return torch.tensor(X, device=self._device, dtype=torch.float)

    def _initialize_predictor(self):
        self._predictor = NearestNeighborPredictorFactory().create(
            'euclidean',
            self._X_train, self._y_train
        )

    def __call__(self, x_eval: np.array, y_eval: int) -> np.array:

        if self.predict_individual(self._new_tensor(x_eval)) != y_eval:
            return np.zeros_like(x_eval)

        else:
            return self._best_perturbation_for_correctly_classified(
                x_eval, y_eval
            )

    def predict_individual(self, x_eval: Tensor):
        return self._predictor.predict_individual(x_eval)

    def _best_perturbation_for_correctly_classified(
            self, x_eval: np.array, y_eval: int
    ):
        dtype = x_eval.dtype
        x_eval = self._new_tensor(x_eval)

        X_pos, X_neg = self._partition_pos_neg(x_eval, y_eval)
        X_screen = self._pos_for_screen(x_eval, X_pos)

        lower_bounds = self._lower_bounds(x_eval, X_neg, X_screen)

        neg_indices = lower_bounds.argsort()

        cache_book = self._cache_book(X_pos)

        best_perturbation = None
        min_perturbation_norm = math.inf

        # count = 0

        for j in neg_indices:

            # print(min_perturbation_norm, lower_bounds[j].item())
            if min_perturbation_norm <= lower_bounds[j]:
                break
            # test
            # count += 1
            perturbation = self._solve_subproblem(
                x_eval, X_neg[j], X_pos,
                cache_book
            )
            perturbation_norm = perturbation.norm().item()
            if perturbation_norm < min_perturbation_norm:
                min_perturbation_norm = perturbation_norm
                best_perturbation = perturbation

        # test
        # print(f'count: {count}')
        # assert self._on_boundary(x_eval + best_perturbation, y_eval)
        return best_perturbation.cpu().numpy().astype(dtype)

    def _partition_pos_neg(self, x_eval: Tensor, y_eval: int):
        mask = (self._y_train == y_eval)
        X_pos = self._X_train[mask]
        X_neg = self._X_train[~mask]
        return X_pos, X_neg

    def _pos_for_screen(self, x_eval: Tensor, X_pos: Tensor):
        if self._screening_check_size is None:
            return X_pos
        else:
            return self._pos_for_screen_helper(x_eval, X_pos)

    def _pos_for_screen_helper(self, x_eval: Tensor, X_pos: Tensor):

        if self._screening_check_size > X_pos.shape[0]:
            raise Exception('The screening check size is too large!')

        nearest_index = self._nearest_pos_index(x_eval, X_pos)
        rand_indices = self._random_pos_indices(X_pos)
        indices = torch.cat((nearest_index.unsqueeze(0), rand_indices))

        return X_pos[indices]

    def _nearest_pos_index(self, x_eval: Tensor, X_pos: Tensor):
        return torch.argmin(torch.norm(x_eval - X_pos, dim=1))

    def _random_pos_indices(self, X_pos: Tensor):
        return torch.randperm(
            X_pos.shape[0], device=X_pos.device
        )[:self._screening_check_size - 1]

    def _cache_book(self, X_pos: Tensor):
        pos_ip = X_pos @ X_pos.t()
        return {
            'pos_ip': pos_ip,
            'pos_squared_norms': pos_ip.diag()
        }

    def _lower_bounds(
            self, x_eval: Tensor, X_neg: Tensor, X_screen: Tensor
    ) -> Tensor:

        # memory for A
        MEMORY_SIZE = 3 * 10**9

        split_size = MEMORY_SIZE // (
            8 * X_screen.shape[0] * x_eval.shape[0]
        )

        if split_size == 0:
            raise Exception('There is not enough memory!')

        X_neg_chunks = torch.split(
            X_neg,
            split_size
        )

        return torch.cat([
            self._lower_bounds_helper(x_eval, X_neg_chunk, X_screen)
            for X_neg_chunk in X_neg_chunks
        ])

    def _lower_bounds_helper(
            self, x_eval: Tensor, X_neg: Tensor, X_screen: Tensor
    ) -> Tensor:

        # It is not necessary to use A_full and b_full for screening
        A, b = self._batch_unbounded_qp_params(
            x_eval, X_neg, X_screen, cache_book={}
        )
        return ((-b).clamp(min=0) / A.norm(dim=2)).max(dim=1)[0]

    def _batch_unbounded_qp_params(
            self, x_eval: Tensor, X_neg: Tensor, X_pos: Tensor,
            cache_book: dict
    ):

        if 'pos_squared_norms' in cache_book:
            pos_squared_norms = cache_book['pos_squared_norms']

        else:
            pos_squared_norms = (X_pos**2).sum(dim=1)

        A = 2 * (X_pos - X_neg.unsqueeze(1))
        b = (
            pos_squared_norms
            - 2 * X_pos @ x_eval
            + ((2 * x_eval - X_neg) * X_neg).sum(dim=1).unsqueeze(1)
        )
        return A, b

    def _solve_subproblem(
        self, x_eval: Tensor, x_neg: Tensor, X_pos: Tensor,
        cache_book: dict
    ) -> Tensor:
        """ Solve the subproblem exactly

        This code is very verbose for efficiency.
        """

        # Q = A @ A.t()
        A, b, Q = self._qp_params(
            x_eval, x_neg, X_pos, cache_book
        )
        lamda = self._qp_solver(Q, b)

        return -A.t() @ lamda

    def _qp_params(
            self, x_eval: Tensor,
            x_neg: Tensor, X_pos: Tensor,
            cache_book: dict
    ):

        A, b, Q = self._unbounded_qp_params(
            x_eval, x_neg, X_pos, cache_book
        )

        if not self._bounded:
            return A, b, Q
        else:
            return self._unbounded_to_bounded_params(
                x_eval, A, b, Q
            )

    def _unbounded_qp_params(
            self, x_eval, x_neg, X_pos, cache_book
    ):
        A, b = self._batch_unbounded_qp_params(
            x_eval, x_neg.unsqueeze(0), X_pos, cache_book
        )
        A, b = A.squeeze(0), b.squeeze(0)
        Q = self._unbounded_Q(x_neg, X_pos, cache_book)
        return A, b, Q

    def _unbounded_Q(
            self, x_neg: Tensor, X_pos: Tensor,
            cache_book
    ):
        # Q = A @ A.t()
        if 'pos_ip' in cache_book:
            pos_ip = cache_book['pos_ip']

        else:
            pos_ip = X_pos @ X_pos.t()

        # A @ A.T = 4 * (X @ X.T - X @ y - (X @ y).T + y.T @ y)
        temp = X_pos @ x_neg
        Q = 4 * (pos_ip - temp.unsqueeze(0)
                 - temp.unsqueeze(1) + x_neg @ x_neg)
        return Q

    def _unbounded_to_bounded_params(
        self, x_eval: Tensor, A: Tensor, b: Tensor, Q: Tensor
    ):

        A1 = torch.eye(x_eval.shape[0], dtype=A.dtype, device=A.device)
        A2 = - torch.eye(x_eval.shape[0], dtype=A.dtype, device=A.device)
        b1 = self._upper - x_eval
        b2 = x_eval - self._lower

        A_full = torch.cat((A, A1, A2))
        b_full = torch.cat((b, b1, b2))

        Q_full = torch.cat(
            (
                torch.cat((Q, A.t(), -A.t())),
                torch.cat((A, A1, A2)),
                torch.cat((-A, A2, A1))
            ),
            dim=1
        )
        return A_full, b_full, Q_full

    def _on_boundary(self, x_eval, y_eval):
        return self._predictor.on_boundary(x_eval, y_eval)


class ExactMahalanobisSolver(ExactEuclideanSolver):

    def __init__(
            self, X_train: np.array, y_train: np.array,
            psd_matrix: np.array,
            qp_solver=QpSolverFactory().create('gcd'),
            screening_check_size=None, bounded=False, upper=1., lower=0.,
            device='cpu', verbose=True
    ):
        self._verbose = verbose
        self._initialize_device(device)

        self._X_train = self._new_tensor(X_train)
        self._y_train = self._new_tensor(y_train)
        self._psd_matrix = self._new_tensor(psd_matrix)

        self._qp_solver = qp_solver
        self._screening_check_size = screening_check_size
        self._bounded = bounded
        self._upper = upper
        self._lower = lower

        self._initialize_predictor()

    def _initialize_predictor(self):
        self._predictor = NearestNeighborPredictorFactory().create(
            'mahalanobis', self._X_train, self._y_train,
            M=self._psd_matrix
        )

    def _cache_book(self, X_pos: Tensor):
        pos_M_ip = (
            X_pos @ self._psd_matrix @ self._psd_matrix.t() @ X_pos.t()
        )
        pos_M_pos_diag = ((X_pos @ self._psd_matrix) * X_pos).sum(dim=1)
        return {
            'pos_M_ip': pos_M_ip,
            'pos_M_pos_diag': pos_M_pos_diag
        }

    def _nearest_pos_index(self, x_eval: Tensor, X_pos: Tensor):
        return torch.argmin(
            (((X_pos - x_eval) @ self._psd_matrix) * (X_pos - x_eval)
             ).sum(dim=1),
        )

    def _batch_unbounded_qp_params(
            self, x_eval: Tensor, X_neg: Tensor, X_pos: Tensor,
            cache_book: dict
    ):

        if 'pos_M_pos_diag' in cache_book:
            pos_M_pos_diag = cache_book['pos_M_pos_diag']

        else:
            pos_M_pos_diag = ((X_pos @ self._psd_matrix) * X_pos).sum(dim=1)

        A = 2 * (X_pos - X_neg.unsqueeze(1)) @ self._psd_matrix
        b = (
            pos_M_pos_diag
            - 2 * X_pos @ self._psd_matrix @ x_eval
            + ((X_neg @ self._psd_matrix) * (2 * x_eval - X_neg)
               ).sum(dim=1).unsqueeze(1)
        )
        return A, b

    def _unbounded_Q(
            self, x_neg: Tensor, X_pos: Tensor,
            cache_book
    ):
        # Q = A @ A.t()
        if 'pos_M_ip' in cache_book:
            pos_M_ip = cache_book['pos_M_ip']

        else:
            raise Exception('Should not be run!')
            pos_M_ip = (
                X_pos @ self._psd_matrix @ self._psd_matrix.t() @ X_pos.t()
            )

        # A @ A.T = 4 * (X @ X.T - X @ y - (X @ y).T + y.T @ y)
        temp = X_pos @ self._psd_matrix @ self._psd_matrix.t() @ x_neg
        Q = 4 * (pos_M_ip - temp.unsqueeze(0)
                 - temp.unsqueeze(1)
                 + (x_neg @ self._psd_matrix) @ (self._psd_matrix.t() @ x_neg))
        return Q

    def _compute_unbounded_qp_params(
            self, x_eval: Tensor, x_neg: Tensor, X_pos: Tensor,
            cache_book: dict = {}, requires_Q: bool = False
    ):

        A = 2 * (X_pos - x_neg) @ self._psd_matrix

        if 'X_pos_M_X_pos_diag' in cache_book:
            X_pos_M_X_pos_diag = cache_book['X_pos_M_X_pos_diag']
        else:
            X_pos_M_X_pos_diag = (
                (X_pos @ self._psd_matrix) * X_pos
            ).sum(dim=1)

        b = (
            X_pos_M_X_pos_diag
            - 2 * X_pos @ self._psd_matrix @ x_eval
            + x_neg @ self._psd_matrix @ (2 * x_eval - x_neg)
        )

        if requires_Q:
            temp = X_pos @ self._psd_matrix @ self._psd_matrix.t() @ x_neg

            if 'X_pos_M_ip' in cache_book:
                X_pos_M_ip = cache_book['X_pos_M_ip']
            else:
                X_pos_M_ip = (
                    X_pos @ self._psd_matrix @ self._psd_matrix.t() @ X_pos.t()
                )

            Q = 4 * (X_pos_M_ip - temp.unsqueeze(0)
                     - temp.unsqueeze(1)
                     + x_neg @ self._psd_matrix @ self._psd_matrix.t() @ x_neg)
        else:
            Q = None

        return A, b, Q


class ExactSolverFactory:

    def create(
            self, name, X_train, y_train, psd_matrix=None,
            qp_solver=QpSolverFactory().create('gcd'),
            screening_check_size=None, bounded=False, upper=1., lower=0.,
            device='cpu', verbose=True
    ):
        if name == 'euclidean':
            result = ExactEuclideanSolver(
                X_train, y_train,
                qp_solver=qp_solver,
                screening_check_size=screening_check_size,
                bounded=bounded, upper=upper, lower=lower,
                device=device, verbose=verbose
            )
        elif name == 'mahalanobis':
            assert psd_matrix is not None
            result = ExactMahalanobisSolver(
                X_train, y_train, psd_matrix,
                qp_solver=qp_solver,
                screening_check_size=screening_check_size,
                bounded=bounded, upper=upper, lower=lower,
                device=device, verbose=verbose
            )
        else:
            raise Exception('unsupported solver')

        return result

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import numpy as np

from typing import Optional, Callable, Any
from typing import Tuple

from metric_robustness.utils.torch_predictors \
    import MahalanobisNearestNeighborPredictor

from metric_robustness.utils.torch_predictors \
    import MahalanobisKnnPredictor

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
        X: Tensor, Y: Tensor, M: Tensor
):

    X_product = ((X @ M) * X).sum(dim=1)
    Y_product = ((Y @ M) * Y).sum(dim=1)
    cross_product = X @ M @ Y.t()
    return (
        X_product.unsqueeze(1)
        + Y_product.unsqueeze(0)
        - 2 * cross_product
    )


class HingeLoss(nn.Module):

    def __init__(self, calibration):
        super().__init__()
        self._calibration = calibration

    def forward(self, margins):
        return (self._calibration - margins).clamp(min=0)


class NegLoss(nn.Module):

    def forward(self, margins):
        return (-margins)


class ExpLoss(nn.Module):

    def forward(self, margins):
        return (-margins).exp()


class LogisticLoss(nn.Module):

    def forward(self, margins):
        return (-margins).exp().log1p()


class CubeNet(nn.Module):

    def __init__(
            self, n_dim, rank: Optional[int] = None,
    ):
        super().__init__()
        if rank is None:
            rank = n_dim

        self.G = torch.nn.Parameter(torch.eye(rank, n_dim))

    def forward(self, X, y):
        M = self.G.t() @ self.G

        csmd = cross_squared_mahalanobis_distances(X, X, M)
        # diff = csmd.unsqueeze(1) - csmd.unsqueeze(2)
        diff = (csmd.unsqueeze(1) - csmd.unsqueeze(2)).clamp(min=0)

        # Is it necessary? Yes!
        # numerator = torch.sign(diff) * (diff)**2
        # numerator = torch.sign(diff) * (diff)**2
        # denominator = 4 * cross_squared_mahalanobis_distances(
        #     X, X, M.t() @ M
        # ).unsqueeze(0)

        numerator = diff
        denominator = 2 * cross_squared_mahalanobis_distances(
            X, X, M.t() @ M
        ).clamp(min=1e-6).sqrt().unsqueeze(0)

        assert torch.isnan(denominator).sum().item() == 0

        margins = numerator / denominator.clamp(min=1e-6)

        # broadcasting
        mask = (y.unsqueeze(0) == y.unsqueeze(1))

        margins[
            (~mask | torch.eye(X.shape[0], device=X.device, dtype=torch.bool)
             ).unsqueeze(2).expand(margins.shape)
        ] = 0
        # ] = - float('inf)
        # margins[
        #     (~mask
        #      ).unsqueeze(2).expand(margins.shape)
        # ] = - float('inf')

        inner_max_margines = margins.max(dim=1)[0]
        inner_max_margines[mask] = float('inf')

        return inner_max_margines.min(dim=1)[0]


class CubeLearner:

    def __init__(
            self,
            optimizer: str,
            optimizer_params: dict,
            criterion: str,
            criterion_params: dict,
            rank: Optional[int],
            batch_size: int,
            n_epochs: int,
            device: str,
            seed: Optional[int] = None,
            verbose: bool = True
    ):
        self._verbose = verbose

        self._initialize_seed(seed)
        self._initialize_device(device)
        self._initialize_optimizer_creator(optimizer, optimizer_params)
        self._initialize_criterion(criterion, criterion_params)

        self._rank = rank
        self._batch_size = batch_size
        self._n_epochs = n_epochs

    def _initialize_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _initialize_device(self, device):
        if not torch.cuda.is_available():
            self._device = 'cpu'
        else:
            self._device = device
        if self._verbose:
            print(f'The learner is using {self._device}!')

    def _initialize_optimizer_creator(self, optimizer, optimizer_params):
        params = {}
        if optimizer == 'sgd':
            if optimizer_params.get('lr') is not None:
                params['lr'] = optimizer_params['lr']
            if optimizer_params.get('momentum') is not None:
                params['momentum'] = optimizer_params['momentum']
            if optimizer_params.get('weight_decay') is not None:
                params['weight_decay'] = optimizer_params['weight_decay']
            self._optimizer_creator = lambda weights: optim.SGD(
                weights, **params
            )
        elif optimizer == 'adam':
            self._optimizer_creator = lambda weights: optim.Adam(
                weights, **params
            )
        else:
            raise Exception('Unsupported optimizer!')

    def _initialize_criterion(self, criterion, criterion_params):
        params = {}
        if criterion == 'hinge':
            if criterion_params.get('calibration') is not None:
                params['calibration'] = criterion_params['calibration']
            self._criterion = HingeLoss(**params)

        elif criterion == 'neg':
            self._criterion = NegLoss(**params)

        elif criterion == 'exp':
            self._criterion = ExpLoss(**params)

        elif criterion == 'logistic':
            self._criterion = LogisticLoss(**params)

        else:
            raise Exception('Unsupported criterion!')

    def __call__(
        self, X, y,
        val_instances: np.array = None, val_labels: np.array = None
    ):

        dtype = X.dtype

        X = torch.tensor(X, dtype=torch.float, device=self._device)
        y = torch.tensor(y, dtype=torch.long, device=self._device)

        if val_instances is not None and val_labels is not None:
            val_instances = torch.tensor(
                val_instances, dtype=torch.float, device=self._device
            )
            val_labels = torch.tensor(
                val_labels, dtype=torch.long, device=self._device
            )

        net = CubeNet(X.shape[1], self._rank).to(self._device)

        optimizer = self._optimizer_creator(net.parameters())

        for i in range(self._n_epochs):

            perm = torch.randperm(X.shape[0], device=self._device)
            X = X[perm]
            y = y[perm]

            train_loss = 0
            # max_loss = -float('inf')
            # min_loss = float('inf')

            for X_mini, y_mini in zip(
                    X.split(self._batch_size), y.split(self._batch_size)
            ):

                margins = net(X_mini, y_mini)
                loss = self._criterion(margins)
                mean_loss = loss.mean()

                # test
                # print(f'margin mean: {margins.mean()}')

                train_loss += mean_loss.item() * X_mini.shape[0]

                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()

                # test
                # print(net.G.grad)

                # test
                # batch_max_loss = loss.max().item()
                # if batch_max_loss > max_loss:
                #     max_loss = batch_max_loss

                # batch_min_loss = loss.min().item()
                # if batch_min_loss < min_loss:
                #     min_loss = batch_min_loss

            if self._verbose:

                print(f'{i:{len(str(self._n_epochs-1))}d}/{self._n_epochs-1}: \
                    {train_loss / X.shape[0]:8.3f}', end='')

                if val_instances is not None and val_labels is not None:
                    with torch.no_grad():
                        M = net.G.t() @ net.G
                        predictor = MahalanobisNearestNeighborPredictor(
                            X, y, M)
                        val_error = 1 - \
                            predictor.score(val_instances, val_labels)
                        print('{:8.3f}'.format(val_error), end='')

                print()

                # test
                # print(f'max loss: {max_loss}')
                # print(f'min loss: {min_loss}')

        return (net.G.t() @ net.G).detach().cpu().numpy().astype(dtype)


class TripleNet(nn.Module):

    def __init__(
            self, n_dim: int,
            random_initialization: bool
    ):
        super().__init__()
        if not random_initialization:
            self.G = nn.Parameter(torch.eye(n_dim))
        else:
            self.G = nn.Parameter(torch.eye(n_dim) + torch.randn(n_dim, n_dim))

    def forward(self, Z, X, Y):
        M = self.G.t() @ self.G
        numerator = ((Z - Y) @ M * (Z - Y)).sum(dim=1) - \
            ((Z - X) @ M * (Z - X)).sum(dim=1)
        denominator = 2 * ((X - Y) @ M.t() @ M * (X - Y)
                           ).sum(dim=1).clamp(min=1e-6).sqrt()
        return numerator / denominator


class TripleLearner:

    def __init__(
            self,
            optimizer: str,
            optimizer_params: dict,
            criterion: str,
            criterion_params: dict,
            n_epochs: int,
            batch_size: Optional[int],
            random_initialization: bool,
            update_triple: bool,
            device: str,
            seed: Optional[int] = None,
            verbose: bool = True,
    ):
        self._verbose = verbose

        self._initialize_seed(seed)
        self._initialize_device(device)
        self._initialize_optimizer_creator(optimizer, optimizer_params)
        self._initialize_criterion(criterion, criterion_params)

        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._random_initialization = random_initialization
        self._update_triple = update_triple

    def _initialize_seed(self, seed: Optional[int]):
        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _initialize_device(self, device: str):
        if not torch.cuda.is_available():
            self._device = 'cpu'
        else:
            self._device = device
        if self._verbose:
            print(f'The learner is using {self._device}!')

    def _initialize_optimizer_creator(
            self, optimizer: str, optimizer_params: dict
    ):
        params = {}
        self._optimizer_creator: Callable[[Any], Any]
        if optimizer == 'sgd':
            if optimizer_params.get('lr') is not None:
                params['lr'] = optimizer_params['lr']
            if optimizer_params.get('momentum') is not None:
                params['momentum'] = optimizer_params['momentum']
            if optimizer_params.get('weight_decay') is not None:
                params['weight_decay'] = optimizer_params['weight_decay']
            self._optimizer_creator = lambda weights: optim.SGD(
                weights, **params
            )
        elif optimizer == 'adam':
            self._optimizer_creator = lambda weights: optim.Adam(
                weights, **params
            )
        else:
            raise Exception('Unsupported optimizer!')

    def _initialize_criterion(self, criterion: str, criterion_params: dict):
        params = {}
        self._criterion: nn.Module
        if criterion == 'hinge':
            if criterion_params.get('calibration') is not None:
                params['calibration'] = criterion_params['calibration']
            self._criterion = HingeLoss(**params)

        elif criterion == 'neg':
            self._criterion = NegLoss()

        elif criterion == 'exp':
            self._criterion = ExpLoss()

        elif criterion == 'logistic':
            self._criterion = LogisticLoss()

        else:
            raise Exception('Unsupported criterion!')

    def __call__(
            self, instances: np.array, labels: np.array,
            val_instances: np.array = None, val_labels: np.array = None,
            val_k: int = 1,
            n_candidate_mins: int = 1, 
    ):
        dtype = instances.dtype

        instances = torch.tensor(
            instances, dtype=torch.float, device=self._device
        )
        labels = torch.tensor(
            labels, dtype=torch.int, device=self._device
        )

        if val_instances is not None and val_labels is not None:
            val_instances = torch.tensor(
                val_instances, dtype=torch.float, device=self._device
            )
            val_labels = torch.tensor(
                val_labels, dtype=torch.int, device=self._device
            )

        net = TripleNet(
            instances.shape[1],
            random_initialization=self._random_initialization
        ).to(self._device)

        optimizer = self._optimizer_creator(net.parameters())

        if self._batch_size is None:
            batch_size = instances.shape[0]
        else:
            batch_size = self._batch_size

        Z, X, Y = TripleLearner._transform_to_triple(instances, labels, n_candidate_mins=n_candidate_mins)

        for i in range(self._n_epochs):
            perm = torch.randperm(Z.shape[0], device=self._device)
            Z = Z[perm]
            X = X[perm]
            Y = Y[perm]

            total_loss = 0.
            for Z_batch, X_batch, Y_batch in zip(
                    Z.split(batch_size),
                    X.split(batch_size),
                    Y.split(batch_size),
            ):
                margins = net(Z_batch, X_batch, Y_batch)
                loss = self._criterion(margins).mean()
                total_loss += loss.item() * Z_batch.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self._verbose:
                print(f'{i:{len(str(self._n_epochs-1))}d}/{self._n_epochs-1}: \
                    {total_loss / Z.shape[0]:8.3f}', end='')

                if val_instances is not None and val_labels is not None:
                    with torch.no_grad():
                        M = net.G.t() @ net.G
                        predictor = MahalanobisKnnPredictor(
                            instances, labels, val_k, M)
                        val_error = 1 - \
                            predictor.score(val_instances, val_labels)
                        print('{:8.3f}'.format(val_error), end='')

                print()

            if self._update_triple and i < self._n_epochs - 1:
                with torch.no_grad():
                    M = net.G.t() @ net.G

                Z, X, Y = TripleLearner._transform_to_triple(
                    instances, labels, psd_matrix=M, n_candidate_mins=n_candidate_mins
                )

        return (net.G.t() @ net.G).detach().cpu().numpy().astype(dtype)

    @staticmethod
    def _transform_to_triple(
            instances: Tensor, labels: Tensor,
            psd_matrix: Optional[Tensor] = None,
            n_candidate_mins: int = 1
    ):
        """return the instances,
        their nearest same-label instances
        and their nearest different-label instances"""

        BATCH_SIZE = 1000

        bias = 0
        Zs = []
        Xs = []
        Ys = []

        while bias < instances.shape[0]:
            Z, X, Y = TripleLearner._transform_to_triple_helper(
                instances, labels, bias, BATCH_SIZE, psd_matrix, n_candidate_mins
            )
            Zs.append(Z)
            Xs.append(X)
            Ys.append(Y)
            bias = bias + BATCH_SIZE

        return torch.cat(Zs), torch.cat(Xs), torch.cat(Ys)

    @staticmethod
    def _transform_to_triple_helper(
            instances: Tensor, labels: Tensor, bias: int, batch_size: int,
            psd_matrix: Optional[Tensor],
            n_candidate_mins: int
    ):
        batch_instances = instances[bias: bias+batch_size]
        batch_labels = labels[bias: bias+batch_size]

        if psd_matrix is None:
            cross_distances = cross_squared_euclidean_distances(
                batch_instances, instances
            )
        else:
            cross_distances = cross_squared_mahalanobis_distances(
                batch_instances, instances, psd_matrix
            )

        same_label_mask = batch_labels.unsqueeze(1) == labels.squeeze(0)

        square_identity_mask = torch.eye(
            batch_instances.shape[0],
            dtype=torch.bool, device=instances.device
        )
        identity_mask = torch.zeros(
            batch_instances.shape[0], instances.shape[0],
            dtype=torch.bool, device=instances.device
        )
        identity_mask[:, bias:bias+batch_size] = square_identity_mask

        if n_candidate_mins == 1:
            _, pos_indices = TripleLearner._conditional_min(
                cross_distances, same_label_mask & ~identity_mask)
            _, neg_indices = TripleLearner._conditional_min(
                cross_distances, ~same_label_mask & ~identity_mask)
        else:
            pos_indices = TripleLearner._conditional_random_min(
                cross_distances, same_label_mask & ~identity_mask, n_candidate_mins)
            neg_indices = TripleLearner._conditional_random_min(
                cross_distances, ~same_label_mask & ~identity_mask, n_candidate_mins)

        Z = batch_instances
        X = torch.index_select(instances, dim=0, index=pos_indices)
        Y = torch.index_select(instances, dim=0, index=neg_indices)

        # is it necessary?
        # if filter:
        #     correct_mask = pos_distances < neg_distances
        #     return (
        #         Z[correct_mask],
        #         X[correct_mask],
        #         Y[correct_mask]
        #     )
        # else:
        return Z, X, Y

    @staticmethod
    def _conditional_min(A: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        A_clone = A.clone()
        A_clone[~mask] = float('inf')
        return A_clone.min(dim=1)

    @staticmethod
    def _conditional_random_min(A: Tensor, mask: Tensor, n_candidates: int) -> Tensor:
        A_clone = A.clone()
        A_clone[~mask] = float('inf')
        _, top_indices = A_clone.topk(
            n_candidates, dim=-1, largest=False, sorted=False)
        rand_indices = torch.randint(
            0, n_candidates, (A.shape[0], 1), device=A.device)
        return top_indices.gather(dim=1, index=rand_indices).squeeze(dim=1)

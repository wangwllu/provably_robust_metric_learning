import torch
import torch.nn as nn
import torch.optim as optim


def cross_squared_mahalanobis_distances(
        X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor
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
        self.calibration = calibration

    def forward(self, margins):
        return (self.calibration - margins).clamp(min=0).mean()


class PairNet(nn.Module):

    def __init__(self, n_dim):
        super().__init__()
        self.G = torch.nn.Parameter(torch.eye(n_dim, n_dim))
        # self.G = torch.nn.Parameter(
        #     torch.eye(n_dim, n_dim) + torch.randn(n_dim, n_dim)
        # )

    def forward(self, X, y):
        M = self.G.t() @ self.G

        # numerator = cross_squared_mahalanobis_distances(X, X, M)
        # denominator = 2 * torch.sqrt(
        #     cross_squared_mahalanobis_distances(X, X, M.t() @ M)
        # )

        # Note for reasonable grad, it has to be squared.
        numerator = cross_squared_mahalanobis_distances(X, X, M) ** 2
        denominator = 4 * cross_squared_mahalanobis_distances(X, X, M.t() @ M)

        margins = numerator / denominator.clamp(min=1e-6)
        mask = y.unsqueeze(0) == y.unsqueeze(1)
        margins[mask] = float('inf')
        return margins.min(dim=1)[0]


class PairLearner:
    """does not work"""

    def __init__(
            self, loss_calibration=20,
            max_iter=1000, learning_rate=1,
            verbose=True
    ):
        self._loss_calibration = loss_calibration
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._verbose = verbose

    def __call__(self, X, y):

        dtype = X.dtype

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        net = PairNet(X.shape[1])
        criterion = HingeLoss(self._loss_calibration)
        optimizer = optim.SGD(net.parameters(), lr=self._learning_rate)

        for i in range(self._max_iter):

            margins = net(X, y)
            loss = criterion(margins)

            if self._verbose:
                print(f'{i:{len(str(self._max_iter-1))}d}/{self._max_iter-1}: \
                     {loss.item():8.4f}')

            optimizer.zero_grad()
            loss.backward()

            # test
            # print(net.G.grad)

            optimizer.step()

        return (net.G.t() @ net.G).detach().cpu().numpy().astype(dtype)


class TripleNet(nn.Module):

    def __init__(self, n_dim):
        super().__init__()
        self.G = torch.nn.Parameter(torch.eye(n_dim, n_dim))
        # self.G = torch.nn.Parameter(
        #     torch.eye(n_dim, n_dim) + torch.randn(n_dim, n_dim)
        # )

    def forward(self, X, y):
        M = self.G.t() @ self.G

        csmd = cross_squared_mahalanobis_distances(X, X, M)
        diff = csmd.unsqueeze(1) - csmd.unsqueeze(2)

        # Is it necessary? Yes!
        numerator = torch.sign(diff) * (diff)**2
        denominator = 4 * cross_squared_mahalanobis_distances(
            X, X, M.t() @ M
        ).unsqueeze(0)

        margins = numerator / denominator.clamp(min=1e-6)

        # show broadcasting

        mask = (y.unsqueeze(0) == y.unsqueeze(1))
        margins[
            (~mask | torch.eye(X.shape[0], device=X.device, dtype=torch.bool)
             ).unsqueeze(2).expand(margins.shape)
        ] = - float('inf')

        inner_max_margines = margins.max(dim=1)[0]
        inner_max_margines[mask] = float('inf')

        return inner_max_margines.min(dim=1)[0]

        # numerator = cross_squared_mahalanobis_distances(X, X, M)
        # denominator = 2 * torch.sqrt(
        #     cross_squared_mahalanobis_distances(X, X, M.t() @ M)
        # )

        # Note for reasonable grad, it has to be squared.


class TripleLearner:

    def __init__(
            self, loss_calibration=20,
            max_iter=100, learning_rate=1, device='cpu',
            verbose=True
    ):
        self._loss_calibration = loss_calibration
        self._max_iter = max_iter
        self._learning_rate = learning_rate

        if not torch.cuda.is_available():
            self._device = 'cpu'
        else:
            self._device = device

        self._verbose = verbose

        if self._verbose:
            print(f'The learner is using {self._device}!')

    def __call__(self, X, y):

        dtype = X.dtype

        X = torch.tensor(X, dtype=torch.float).to(self._device)
        y = torch.tensor(y, dtype=torch.float).to(self._device)

        net = TripleNet(X.shape[1]).to(self._device)
        criterion = HingeLoss(self._loss_calibration).to(self._device)

        optimizer = optim.SGD(net.parameters(), lr=self._learning_rate)

        for i in range(self._max_iter):

            margins = net(X, y)
            loss = criterion(margins)

            # test
            # print(f'margin mean: {margins.mean()}')

            if self._verbose:
                print(f'{i:{len(str(self._max_iter-1))}d}/{self._max_iter-1}: \
                     {loss.item():8.4f}')

            optimizer.zero_grad()
            loss.backward()

            # test
            # print(net.G.grad)

            optimizer.step()

        return (net.G.t() @ net.G).detach().cpu().numpy().astype(dtype)


class MiniBatchTripleLearner(TripleLearner):

    def __init__(
            self, loss_calibration=20,
            max_iter=100, learning_rate=1, device='cpu',
            batch_size=300,
            verbose=True,
    ):
        super().__init__(
            loss_calibration, max_iter, learning_rate,
            device, verbose
        )
        self._batch_size = 300

    def __call__(self, X, y):

        dtype = X.dtype

        X = torch.tensor(X, dtype=torch.float).to(self._device)
        y = torch.tensor(y, dtype=torch.float).to(self._device)

        net = TripleNet(X.shape[1]).to(self._device)
        criterion = HingeLoss(self._loss_calibration).to(self._device)

        optimizer = optim.SGD(net.parameters(), lr=self._learning_rate)

        for i in range(self._max_iter):

            train_loss = 0

            for X_mini, y_mini in zip(
                    X.split(self._batch_size), y.split(self._batch_size)
            ):

                margins = net(X_mini, y_mini)
                loss = criterion(margins)

                # test
                # print(f'margin mean: {margins.mean()}')

                train_loss += loss.item() * X_mini.shape[0]

                optimizer.zero_grad()
                loss.backward()

                # test
                # print(net.G.grad)

                optimizer.step()

            if self._verbose:
                print(f'{i:{len(str(self._max_iter-1))}d}/{self._max_iter-1}: \
                    {train_loss / X.shape[0]:8.4f}')

        return (net.G.t() @ net.G).detach().cpu().numpy().astype(dtype)

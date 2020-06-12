import torch
from torch import Tensor
from typing import Tuple


def triple_verify(z: Tensor, x: Tensor, y: Tensor) -> float:
    """Solve the triplet problem with respect to only one single z, x, and y

    This function is not used in practice."""

    numerator = (((z - y)**2).sum() - ((z - x)**2).sum()).clamp(min=0)
    denominator = 2 * ((y - x)**2).sum().sqrt().clamp(min=1e-6)
    return (numerator / denominator).item()


def batch_triple_verify(z: Tensor, X: Tensor, Y: Tensor, k: int) -> float:
    s = (k + 1) // 2
    BATCH_SIZE = 100

    vertical_epsilons_list = []
    for Y_batch in torch.split(Y, BATCH_SIZE):

        horizontal_epsilons_list = []
        for X_batch in torch.split(X, BATCH_SIZE):
            horizontal_epsilons_list.append(
                batch_triple_verify_helper(z, X_batch, Y_batch))
        horizontal_epsilons = torch.cat(horizontal_epsilons_list, dim=1)

        vertical_epsilons_list.append(
            horizontal_epsilons.kthvalue(
                1 + horizontal_epsilons.shape[1] - s, dim=1
            )[0]
        )
    vertical_epsilons = torch.cat(vertical_epsilons_list, dim=0)
    return vertical_epsilons.kthvalue(
        s, dim=0,
    )[0].item()


def batch_triple_verify_helper(z: Tensor, X: Tensor, Y: Tensor) -> Tensor:
    numerator = (((z - Y)**2).sum(dim=1).unsqueeze(1) -
                 ((z - X)**2).sum(dim=1).unsqueeze(0)).clamp(min=0)
    denominator = 2 * ((Y.unsqueeze(1) - X.unsqueeze(0)) **
                       2).sum(dim=-1).sqrt().clamp(min=1e-6)
    return numerator / denominator

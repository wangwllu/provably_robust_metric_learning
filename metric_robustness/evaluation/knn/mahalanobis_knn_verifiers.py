import torch
from torch import Tensor
from typing import Tuple, Optional


def triple_verify(z: Tensor, x: Tensor, y: Tensor, M: Tensor) -> float:
    """Solve the triplet problem with respect to only one single z, x, and y

    This function is not used in practice."""

    numerator = ((z - y) @ M @ (z - y) - (z - x) @ M @ (z - x)).clamp(min=0)
    denominator = 2 * ((y - x) @ M @ M @ (y - x)).sqrt().clamp(min=1e-6)

    return (numerator / denominator).item()


def batch_triple_verify_helper(
        z: Tensor, X: Tensor, Y: Tensor, M: Tensor,
        cache_X_M: Optional[Tensor] = None,
        cache_Y_M: Optional[Tensor] = None
) -> Tensor:

    if cache_X_M is not None:
        X_M = cache_X_M
    else:
        X_M = X @ M

    if cache_Y_M is not None:
        Y_M = cache_Y_M
    else:
        Y_M = Y @ M

    numerator = (((z @ M - Y_M) * (z - Y)).sum(dim=1).unsqueeze(1) -
                 ((z @ M - X_M) * (z - X)).sum(dim=1).unsqueeze(0)).clamp(min=0)

    denominator = 2 * ((Y_M.unsqueeze(1) - X_M.unsqueeze(0))
                       ** 2).sum(dim=-1).sqrt().clamp(min=1e-6)

    return numerator / denominator


def batch_triple_verify(z: Tensor, X: Tensor, Y: Tensor, k: int, M: Tensor) -> float:
    s = (k + 1) // 2
    BATCH_SIZE = 100

    X_M = X @ M
    Y_M = Y @ M

    vertical_epsilons_list = []
    for Y_M_batch, Y_batch in zip(torch.split(Y_M, BATCH_SIZE), torch.split(Y, BATCH_SIZE)):

        horizontal_epsilons_list = []
        for X_M_batch, X_batch in zip(torch.split(X_M, BATCH_SIZE), torch.split(X, BATCH_SIZE)):
            horizontal_epsilons_list.append(
                batch_triple_verify_helper(z, X_batch, Y_batch, M, X_M_batch, Y_M_batch))
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

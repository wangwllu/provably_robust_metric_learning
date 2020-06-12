"""add a perturbation delta on z such that z + delta is closer to y than x"""

import math
from torch import Tensor


def triplet(z: Tensor, x: Tensor, y: Tensor):
    a = 2 * (x - y)
    b = x @ x - y @ y - 2 * (x - y) @ z

    return qp_one_constraint(a, b)


# not very useful, since we have triplet_batch_X_Y
def triplet_batch_X(z: Tensor, X: Tensor, y: Tensor):
    a = 2 * (X - y)
    b = (X**2).sum(dim=1) - y @ y - 2 * (X - y) @ z

    return qp_one_constraint_batch(a, b)


def triplet_batch_X_Y(z: Tensor, X: Tensor, Y: Tensor):
    a = 2 * (X - Y.unsqueeze(1))
    b = (
        (X**2).sum(dim=1) - (Y**2).sum(dim=1).unsqueeze(1)
        - 2 * (X @ z - (Y @ z).unsqueeze(1))
    )

    return qp_one_constraint_batch(a, b)


def triplet_psd(z: Tensor, x: Tensor, y: Tensor, M: Tensor):
    a = 2 * (x - y) @ M
    b = x @ M @ x - y @ M @ y - 2 * (x - y) @ M @ z

    return qp_one_constraint(a, b)


def triplet_psd_batch_X_Y(z: Tensor, X: Tensor, Y: Tensor, M: Tensor):
    a = 2 * (X @ M - (Y @ M).unsqueeze(1))
    b = (
        ((X @ M) * X).sum(dim=1) - ((Y @ M) * Y).sum(dim=1).unsqueeze(1)
        - 2 * (X @ M @ z - (Y @ M @ z).unsqueeze(1))
    )

    return qp_one_constraint_batch(a, b)


def qp_one_constraint(a: Tensor, b: Tensor):
    """solve min ||x|| s.t. a^T x <= b

    return both the optimal point and the optimal value
    """

    opt_point = - (-b).clamp(min=0) * a / (a @ a).clamp(1e-6)
    opt_value = (-b).clamp(min=0) / a.norm().clamp(1e-6)

    return opt_point, opt_value


def qp_one_constraint_batch(a: Tensor, b: Tensor):
    """solve a batch of problems min ||x|| s.t. a^T x <= b

    a: [*shape,  d]
    b: [*shape]
    only return optimal values
    """

    opt_values = (-b).clamp(min=0) / a.norm(dim=-1).clamp(min=1e-6)
    return opt_values


def kernel_difference(
        z: Tensor, x: Tensor, y: Tensor,
        W: Tensor, M: Tensor, delta: Tensor
):
    # temp = ((W @ (z + delta)).relu() @ M @ (W @ x).relu()
    #         - (W @ (z + delta)).relu() @ M @ (W @ y).relu())

    v = ((W @ x).relu() - (W @ y).relu()) @ M
    result = v @ (W @ (z + delta)).relu()

    # print(result)
    # print(temp)

    # assert torch.allclose(temp, result)
    return result


def linear_lower_bound(
        z: Tensor, x: Tensor, y: Tensor,
        W: Tensor, M: Tensor, delta: Tensor, epsilon: float
):
    upper = W @ z + W.norm(dim=1) * epsilon
    lower = W @ z - W.norm(dim=1) * epsilon
    ratio = upper / (upper - lower).clamp(min=1e-6)

    v = ((W @ x).relu() - (W @ y).relu()) @ M

    pos_mask = (lower >= 0)
    uns_mask = (lower < 0) & (upper > 0)
    reg_mask = (lower < 0) & (upper > 0) & (v < 0)

    # temp = (
    #     v[pos_mask] @ (W[pos_mask] @ (z + delta))
    #     + (v[uns_mask] * ratio[uns_mask]) @ (W[uns_mask] @ (z + delta))
    #     - (v[reg_mask] * ratio[reg_mask]) @ lower[reg_mask]
    # )

    result = (
        W[pos_mask].t() @ v[pos_mask] @ (z + delta)
        + W[uns_mask].t() @ (v[uns_mask] * ratio[uns_mask]) @ (z + delta)
        - (v[reg_mask] * ratio[reg_mask]) @ lower[reg_mask]
    )

    # assert torch.allclose(temp, result)

    return result


def triplet_relu(
        z: Tensor, x: Tensor, y: Tensor,
        W: Tensor, M: Tensor, epsilon: float
):

    upper = W @ z + W.norm(dim=1) * epsilon
    lower = W @ z - W.norm(dim=1) * epsilon
    ratio = upper / (upper - lower).clamp(min=1e-6)

    v = ((W @ x).relu() - (W @ y).relu()) @ M

    pos_mask = (lower >= 0)
    uns_mask = (lower < 0) & (upper > 0)
    reg_mask = (lower < 0) & (upper > 0) & (v < 0)

    a = 2 * (
        W[pos_mask].t() @ v[pos_mask]
        + W[uns_mask].t() @ (v[uns_mask] * ratio[uns_mask])
    )

    b = (
        (W @ x).relu() @ M @ (W @ x).relu()
        - (W @ y).relu() @ M @ (W @ y).relu()
    ) - 2 * (
        W[pos_mask].t() @ v[pos_mask] @ z
        + W[uns_mask].t() @ (v[uns_mask] * ratio[uns_mask]) @ z
        - (v[reg_mask] * ratio[reg_mask]) @ lower[reg_mask]
    )

    return qp_one_constraint(a, b)


# TODO: triplet relu batch
# def triplet_relu_batch_X(
#         z: Tensor, X: Tensor, y: Tensor,
#         W: Tensor, M: Tensor, epsilon: Tensor
# ):
#     """ a batch version for triplet_relu

#     z: [D]
#     X: [N, D]
#     y: [D]
#     W: [D, D']
#     M: [D', D']
#     epsilon: [N]
#     """

#     # upper, lower, ratio, pos_mask, uns_mask: [N, D']
#     upper = z @ W + W.norm(dim=0) * epsilon.unsqueeze(-1)
#     lower = z @ W - W.norm(dim=0) * epsilon.unsqueeze(-1)
#     ratio = upper / (upper - lower).clamp(min=1e-6)

#     pos_mask = (lower >= 0)
#     uns_mask = (lower < 0) & (upper > 0)

#     # v, reg_mask: [N, D']
#     v = ((X @ W).relu() - (y @ W).relu()) @ M
#     reg_mask = (lower < 0) & (upper > 0) & (v < 0)

#     a = 2 * (
#         W[pos_mask].t() @ v[pos_mask]
#         + W[uns_mask].t() @ (v[uns_mask] * ratio[uns_mask])
#     )


def triplet_relu_adaptive(
        z: Tensor, x: Tensor, y: Tensor,
        W: Tensor, M: Tensor, initial_epsilon: float = 1.
) -> float:
    upper = None
    epsilon = initial_epsilon

    while True:
        _, opt = triplet_relu(z, x, y, W, M, epsilon)
        opt = opt.item()

        # print(opt - epsilon)

        if math.isclose(opt, epsilon, rel_tol=1e-5, abs_tol=1e-6):
            return opt

        if opt < epsilon:
            upper = epsilon
            epsilon = (epsilon + opt) / 2

        else:
            if upper is not None:
                epsilon = (upper + epsilon) / 2
            else:
                epsilon *= 2

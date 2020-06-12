import numpy as np


def kth_min(a, k):
    return np.partition(a, k-1)[k-1]


def kth_max(a, k):
    idx = a.shape[0] - k
    return np.partition(a, idx)[idx]


def top_k_min_indices(a, k):
    return np.argpartition(a, k-1)[:k]

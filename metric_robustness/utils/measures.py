import numpy as np


def compute_robust_certified_error(perturbation_norms):

    unique_norms, counts = np.unique(perturbation_norms, return_counts=True)

    return unique_norms, np.cumsum(counts) / np.sum(counts)

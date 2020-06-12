import numpy as np
import os
import torch

from metric_robustness.utils.initialization import read_params
from metric_robustness.utils.initialization import initialize_results_dir
from metric_robustness.utils.initialization import backup_params
from metric_robustness.utils.loaders import LoaderFactory
from metric_robustness.utils.torch_predictors import KnnFactory

import metric_robustness.evaluation.knn.mahalanobis_knn_verifiers as maha
import metric_robustness.evaluation.knn.euclidean_knn_verifiers as euc

# from metric_robustness.evaluation.nn_only.exact import ExactSolverFactory


DEFAULT_CONFIG_PATH = 'config/knn_verify.ini'
params = read_params(DEFAULT_CONFIG_PATH)
initialize_results_dir(params.get('results_dir'))
backup_params(params, params.get('results_dir'))


X_train, y_train, X_test, y_test = LoaderFactory().create(
    name=params.get('dataset'),
    root=params.get('dataset_dir'),
    random=True,
    seed=params.getint('split_seed')
)()


X_train = torch.tensor(X_train, dtype=torch.float, device=params.get('device'))
y_train = torch.tensor(y_train, dtype=torch.long, device=params.get('device'))
X_test = torch.tensor(X_test, dtype=torch.float, device=params.get('device'))
y_test = torch.tensor(y_test, dtype=torch.long, device=params.get('device'))


if params.get('metric') == 'euclidean':
    psd_matrix = None

    def verifier(z, X, Y): return euc.batch_triple_verify(
        z, X, Y, params.getint('k'))

elif params.get('metric') == 'mahalanobis':
    psd_matrix = np.loadtxt(
        params.get('psd_matrix_path'),
    )
    psd_matrix = torch.tensor(
        psd_matrix, dtype=torch.float, device=params.get('device'))

    def verifier(z, X, Y): return maha.batch_triple_verify(
        z, X, Y, params.getint('k'), M=psd_matrix
    )

else:
    raise Exception('unsupported metric')


predictor = KnnFactory().create(
    params.get('metric'), X_train, y_train,
    n_neighbors=params.getint('k'), M=psd_matrix
)


n_eval = params.getint('n_eval')
perturbation_norms = np.empty(n_eval)
for i in range(n_eval):

    if predictor.predict_individual(X_test[i]) == y_test[i]:
        mask = (y_train == y_test[i])
        perturbation_norm = verifier(X_test[i], X_train[mask], X_train[~mask])
    else:
        perturbation_norm = 0

    perturbation_norms[i] = perturbation_norm

    print(f'{i+1:{len(str(n_eval))}d}/{n_eval}: {perturbation_norm:8.4f}')

np.savetxt(
    os.path.join(params.get('results_dir'), 'perturbation_norms.txt'),
    perturbation_norms
)

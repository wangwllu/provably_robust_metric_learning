import numpy as np
import os

from metric_robustness.utils.initialization import read_params
from metric_robustness.utils.initialization import initialize_results_dir
from metric_robustness.utils.initialization import backup_params
from metric_robustness.utils.loaders import LoaderFactory
from metric_robustness.evaluation.nn_only.torch_exact import ExactSolverFactory
# from metric_robustness.evaluation.nn_only.exact import ExactSolverFactory


DEFAULT_CONFIG_PATH = 'config/exact_perturbation_norms.ini'
params = read_params(DEFAULT_CONFIG_PATH)
initialize_results_dir(params.get('results_dir'))
backup_params(params, params.get('results_dir'))


X_train, y_train, X_test, y_test = LoaderFactory().create(
    name=params.get('dataset'),
    root=params.get('dataset_dir'),
    random=True,
    seed=params.getint('split_seed')
)()


if params.get('metric') == 'euclidean':
    psd_matrix = None

elif params.get('metric') == 'mahalanobis':
    psd_matrix = np.loadtxt(
        params.get('psd_matrix_path'),
        dtype=X_train.dtype
    )

else:
    raise Exception('unsupported metric')

solver = ExactSolverFactory().create(
    params.get('metric'),
    X_train, y_train,
    psd_matrix=psd_matrix,
    screening_check_size=params.getint('screening_check_size'),
    bounded=params.getboolean('bounded'),
    device=params.get('device', 'cuda'),
)


n_eval = params.getint('n_eval')
perturbation_norms = np.empty(n_eval)
for i in range(n_eval):

    perturbation = solver(X_test[i], y_test[i])
    perturbation_norm = np.linalg.norm(perturbation)
    perturbation_norms[i] = perturbation_norm

    print(f'{i+1:{len(str(n_eval))}d}/{n_eval}: {perturbation_norm:8.4f}')

np.savetxt(
    os.path.join(params.get('results_dir'), 'perturbation_norms.txt'),
    perturbation_norms
)

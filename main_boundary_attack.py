import numpy as np
import os
import torch

from metric_robustness.utils.initialization import read_params
from metric_robustness.utils.initialization import initialize_results_dir
from metric_robustness.utils.initialization import backup_params
from metric_robustness.utils.loaders import LoaderFactory

from metric_robustness.utils.torch_predictors import MahalanobisKnnModule

from foolbox import PyTorchModel
from foolbox.attacks import BoundaryAttack


DEFAULT_CONFIG_PATH = 'config/boundary_attack.ini'
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
    psd_matrix = torch.eye(
        X_train.shape[1], dtype=torch.float, device=params.get('device'))

elif params.get('metric') == 'mahalanobis':
    psd_matrix = np.loadtxt(
        params.get('psd_matrix_path'),
    )
    psd_matrix = torch.tensor(
        psd_matrix, dtype=torch.float, device=params.get('device'))

else:
    raise Exception('unsupported metric')

knn_module = MahalanobisKnnModule(
    X_train, y_train, params.getint('k'), psd_matrix
)
knn_module.to(params.get('device'))
knn_module.eval()

fmodel = PyTorchModel(knn_module, bounds=(0, 1), device=params.get('device'))
attack = BoundaryAttack()

n_eval = params.getint('n_eval')
perturbations_list = []


for i, (X_eval, y_eval) in enumerate(zip(
    torch.split(X_test[:n_eval], params.getint('attack_batch_size')),
    torch.split(y_test[:n_eval], params.getint('attack_batch_size')),
)):
    print(i)
    _, advs, successful = attack(
        fmodel, X_eval, y_eval,
        epsilons=None,
    )
    perturbations = (X_eval - advs).norm(dim=1)
    perturbations[~successful] = float('inf')

    wrongly_predicted = knn_module(X_eval).argmax(dim=1) != y_eval
    perturbations[wrongly_predicted] = 0.

    perturbations_list.append(perturbations)

perturbations = torch.cat(perturbations_list)
np.savetxt(
    os.path.join(params.get('results_dir'), 'perturbation_norms.txt'),
    perturbations.cpu().numpy()
)

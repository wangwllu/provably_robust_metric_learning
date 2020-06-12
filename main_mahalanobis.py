import numpy as np
import os

from metric_robustness.utils.initialization import read_params
from metric_robustness.utils.initialization import initialize_results_dir
from metric_robustness.utils.initialization import backup_params

from metric_robustness.utils.loaders import LoaderFactory
from metric_learn import NCA, LMNN, ITML_Supervised, LFDA

from metric_robustness.learning.verifier_based_learners \
    import TripleLearner


DEFAULT_CONFIG_PATH = 'config/mahalanobis.ini'
params = read_params(DEFAULT_CONFIG_PATH)


def main(params):

    initialize_results_dir(params.get('results_dir'))
    backup_params(params, params.get('results_dir'))

    print('>>> loading data...')

    X_train, y_train, X_test, y_test = LoaderFactory().create(
        name=params.get('dataset'),
        root=params.get('dataset_dir'),
        random=True,
        seed=params.getint('split_seed')
    )()

    print('<<< data loaded')

    print('>>> computing psd matrix...')

    if params.get('algorithm') == 'identity':
        psd_matrix = np.identity(X_train.shape[1], dtype=X_train.dtype)

    elif params.get('algorithm') == 'nca':
        nca = NCA(
            init='auto', verbose=True,
            random_state=params.getint('algorithm_seed')
        )
        nca.fit(X_train, y_train)
        psd_matrix = nca.get_mahalanobis_matrix()

    elif params.get('algorithm') == 'lmnn':
        lmnn = LMNN(
            init='auto', verbose=True,
            random_state=params.getint('algorithm_seed')
        )
        lmnn.fit(X_train, y_train)
        psd_matrix = lmnn.get_mahalanobis_matrix()

    elif params.get('algorithm') == 'itml':
        itml = ITML_Supervised(
            verbose=True,
            random_state=params.getint('algorithm_seed')
        )
        itml.fit(X_train, y_train)
        psd_matrix = itml.get_mahalanobis_matrix()

    elif params.get('algorithm') == 'lfda':

        lfda = LFDA()
        lfda.fit(X_train, y_train)
        psd_matrix = lfda.get_mahalanobis_matrix()

    elif params.get('algorithm') == 'arml':
        learner = TripleLearner(
            optimizer=params.get('optimizer'),
            optimizer_params={
                'lr': params.getfloat('lr'),
                'momentum': params.getfloat('momentum'),
                'weight_decay': params.getfloat('weight_decay'),
            },
            criterion=params.get('criterion'),
            criterion_params={'calibration': params.getfloat('calibration')},
            n_epochs=params.getint('n_epochs'),
            batch_size=params.getint('batch_size'),
            random_initialization=params.getboolean(
                'random_initialization', fallback=False
            ),
            update_triple=params.getboolean('update_triple', fallback=False),
            device=params.get('device'),
            seed=params.getint('learner_seed')
        )

        psd_matrix = learner(
            X_train, y_train,
            n_candidate_mins=params.getint('n_candidate_mins', fallback=1)
        )

    else:
        raise Exception('unsupported algorithm')

    print('<<< psd matrix got')

    np.savetxt(
        os.path.join(params.get('results_dir'), 'psd_matrix.txt'),
        psd_matrix
    )


if __name__ == "__main__":
    main(params)

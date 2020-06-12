import configparser
import argparse
import os

from datetime import datetime
from .loaders import LoaderFactory
from .type_conversion import str_to_int_list


def initialize_params(token, time_stamped=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join('config', f'{token}.ini')
    )
    parser.add_argument(
        '--section',
        default='DEFAULT'
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    params = config[args.section]

    if time_stamped:
        params['result_dir'] = os.path.join(
            params.get('result_dir'),
            '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())
        )

    if not os.path.exists(params.get('result_dir')):
        os.makedirs(params.get('result_dir'))

    with open(
        os.path.join(params.get('result_dir'), 'config.ini'), 'w'
    ) as backup_configfile:
        config.write(backup_configfile)

    return params


def initialize_data(params):
    loader = LoaderFactory().create(
        name=params.get('dataset'),
        root=params.get('dataset_dir'),
        random=params.getboolean('random'),
        seed=params.getint('seed', fallback=None),
        partial=params.getboolean('partial', fallback=False),
        label_domain=str_to_int_list(params.get('label_domain'))
    )

    return loader.load()

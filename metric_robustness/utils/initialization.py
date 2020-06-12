import argparse
import os
import configparser

from datetime import datetime


def read_params(default_config_path):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default=default_config_path
    )
    parser.add_argument('--section', default='DEFAULT')
    args = parser.parse_args()

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(args.config)
    return config[args.section]


def initialize_results_dir(results_dir):
    if os.path.exists(results_dir):
        os.rename(results_dir, _append_time_stamp(results_dir))
    os.makedirs(results_dir)


def backup_params(params: configparser.SectionProxy, results_dir):
    config = configparser.ConfigParser()
    config['DEFAULT'] = params
    with open(
        os.path.join(results_dir, 'backup_config.ini'), 'w'
    ) as config_file:
        config.write(config_file)


def _append_time_stamp(results_dir):
    return results_dir + '_' + _format_time_stamp()


def _format_time_stamp():
    return '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now())

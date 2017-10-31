import yaml
import os
import math
import logging

import numpy as np
from decimal import Decimal
from datetime import datetime


def __return_yaml(file_name):
    wd = os.getcwd()
    setup_path = os.path.join(wd, file_name)
    with open(setup_path, 'r') as stream:
        return yaml.load(stream)

def get_setup(file_name = None):
    return __return_yaml(file_name or 'setup.yml')

def get_grid_search_results():
    return __return_yaml('grid_search_results.yml')

def get_tau_results():
    return __return_yaml('tau_results.yml')

def random_subset_indices(np_array, n_first_subset):
    """ Return a tuple of two indices lists that split np_array randomly.
    First subset length: n_first_subset
    Last subset length: len(np_array) - n_first_subset
    """
    idx = np.arange(0, len(np_array))
    np.random.shuffle(idx)
    idx1 = idx[0:n_first_subset]
    idx2 = idx[n_first_subset:]
    return idx1, idx2


def add_to_collection(sample, samples):
    """Samples is a 3D-array with 
    - MC sample on axis 0
    - input examples by row (if predicting multiple, else only one row)
    - output dimensions by col
    """
    if sample.ndim == 1:
        # 1D output (If predicting 1 example)
        sample = sample.reshape(-1, 1)
    if samples is None:
        return np.array([sample])
    # Else append sample to the 0th axis of samples collection
    return np.append(samples, [sample], axis=0)


def get_lambdas_range(flt_min, flt_max):
    """Convert scientific min/max values with base 10 to list of intermediate
    values by factor 10.
    """
    dec_min = Decimal(flt_min)
    dec_max = Decimal(flt_max)
    exp_min = math.log10(dec_min)
    exp_max = math.log10(dec_max)
    n_lambdas = abs(exp_min) - abs(exp_max) + 1
    return np.logspace(exp_max, exp_min, n_lambdas).tolist()


def __get_unique_path(parent_path, subdir):
    make_path_if_missing(parent_path)
    target_path = os.path.join(parent_path, subdir)

    if os.path.exists(target_path):
        existing_similar = [d for d in os.listdir(parent_path) if d.startswith(subdir)]
        subdir += '_{}'.format(len(existing_similar))

    subdir_path = os.path.join(parent_path, subdir)
    os.makedirs(subdir_path)
    return subdir_path


def get_new_dir_in_parent_path(parent_path, subdir=None):
    """Create a new unique dir in parent_path and return the full path"""
    subdir = subdir or datetime.now().strftime("%Y-%m-%d_%H%M")
    return __get_unique_path(parent_path, subdir)


def make_path_if_missing(path):
    """Make path if it doesn't exist already"""
    try:
        os.makedirs(path)
    except os.error:
        pass

def dump_yaml(dictionary, parent_path, filename):
    file_path = os.path.join(parent_path, filename)
    with file(file_path, 'w') as stream:
        yaml.dump(dictionary, stream, default_flow_style = False)

def get_directories_in_path(path):
    nodes = os.listdir(path)
    return [node for node in nodes if os.path.isdir(os.path.join(path, node))]

def get_train_and_evaluation_models(models_list):
    """Return a dict with trained model names as keys and corresp. evaluated models as values.

    E.g. if models_list is ['MCBN', 'MCDO', 'BN', 'DO'], return (list order does not matter)
    {
        'BN': ['BN', 'MCBN']
        'DO': ['DO', 'MCDO']
    }
    and if models_list is ['MCBN', 'MCDO'], return
    {
        'BN': ['MCBN']
        'DO': ['MCDO']
    }
    since MCBN is trained as a BN model but inference is done uniquely, and correspondingly for MCDO.
    """
    base_models = list(set([m.replace('MC','') for m in models_list]))
    return {bm: [m for m in models_list if bm in m] for bm in base_models}

def get_logger():
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='evaluation.log', mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

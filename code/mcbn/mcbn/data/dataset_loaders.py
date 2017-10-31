import os

import pandas as pd
import numpy as np

from mcbn.utils.helper import make_path_if_missing
from mcbn.utils.helper import get_new_dir_in_parent_path
from mcbn.environment.constants import DATA_PATH


def load_uci_info(name):
    dataset_path = os.path.join(DATA_PATH, name)

    feature_indices_path = os.path.join(dataset_path, 'index_features.txt')
    feature_indices = np.loadtxt(feature_indices_path, dtype=int).tolist()

    target_indices_path = os.path.join(dataset_path, 'index_target.txt')
    target_indices = np.loadtxt(target_indices_path, dtype=int).tolist()

    if type(feature_indices) == int:
        feature_indices = [feature_indices]
    if type(target_indices) == int:
        target_indices = [target_indices]

    return feature_indices, target_indices


def load_uci_data_as_dataframe(name):
    dataset_path = os.path.join(DATA_PATH, name)
    all_files = os.listdir(dataset_path)
    dataset_filename = next(f for f in all_files if f.endswith('.data'))
    dataset_path = os.path.join(dataset_path, dataset_filename)
    return pd.read_csv(dataset_path, engine='python', header=None, delim_whitespace=True)


def load_uci_data_full(name):
    df = load_uci_data_as_dataframe(name)

    feature_indices, target_indices = load_uci_info(name)
    X = df.loc[:, feature_indices].values
    y = df.loc[:, target_indices].values

    # Ensure target arrays are 2 dimensional
    y = y.reshape(-1, 1) if len(y.shape) == 1 else y
    return X, y


def load_fold(folds_path, fold, X_full, y_full):
    train_idx_path = os.path.join(folds_path, '{}_train_idx.txt'.format(fold))
    val_idx_path = os.path.join(folds_path, '{}_val_idx.txt'.format(fold))
    train_idx = np.loadtxt(train_idx_path, dtype=int).tolist()
    val_idx = np.loadtxt(val_idx_path, dtype=int).tolist()

    X_train = X_full[train_idx, :]
    y_train = y_full[train_idx, :]
    X_val = X_full[val_idx, :]
    y_val = y_full[val_idx, :]

    return X_train, y_train, X_val, y_val


def load_uci_data_test(dataset_name):
    X_full, y_full = load_uci_data_full(dataset_name)
    indices_path = os.path.join(DATA_PATH, dataset_name, 'train_cv-test')

    train_idx_path = os.path.join(indices_path, 'train_cv_indices.txt')
    test_idx_path = os.path.join(indices_path, 'test_indices.txt')

    train_idx = np.loadtxt(train_idx_path, dtype=int).tolist()
    test_idx = np.loadtxt(test_idx_path, dtype=int).tolist()

    X_train = X_full[train_idx, :]
    y_train = y_full[train_idx, :]
    X_test = X_full[test_idx, :]
    y_test = y_full[test_idx, :]

    return X_train, y_train, X_test, y_test

def save_indices(path, filename, indices):
    make_path_if_missing(path)
    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as f:
        np.savetxt(f, indices, fmt='%d')


def create_folds(dataset_name, n_folds, inverted_cv_fraction, parent_path):
    train_cv_idx_path = os.path.join(DATA_PATH, dataset_name, 'train_cv-test', 'train_cv_indices.txt')
    train_cv_idx = np.loadtxt(train_cv_idx_path, dtype=int).tolist()
    np.random.shuffle(train_cv_idx)

    val_idx_per_fold = np.array_split(train_cv_idx, inverted_cv_fraction)[:n_folds]
    folds_path = get_new_dir_in_parent_path(parent_path, subdir='fold_indices')

    for i, fold_val_idx in enumerate(val_idx_per_fold):
        # Get (sorted) fold_train_idx and shuffle
        fold_train_idx = np.setdiff1d(train_cv_idx, fold_val_idx)
        np.random.shuffle(fold_train_idx)

        # Save fold
        save_indices(folds_path, '{}_val_idx.txt'.format(i), fold_val_idx)
        save_indices(folds_path, '{}_train_idx.txt'.format(i), fold_train_idx)

    return folds_path

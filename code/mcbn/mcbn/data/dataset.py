import numpy as np

from mcbn.utils.helper import random_subset_indices


class Dataset:
    def __init__(self,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 discard_leftovers,
                 normalize_X=True,
                 normalize_y=False):

        """Feature and target vectors must be np arrays"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0)
        self.y_mean = y_train.mean(axis=0)
        self.y_std = y_train.std(axis=0)

        # Set std dev to 1 for constant features
        self.X_std[np.all(X_train == X_train[0, :], axis=0)] = 1.0

        self.norm_X = normalize_X
        self.norm_y = normalize_y

        # Keep track of epoch train data order
        self.epoch_indices = []
        self.curr_epoch = 0

        # Discard leftover examples at end of epoch
        self.discard_leftovers = discard_leftovers

    def next_batch(self, M):
        """Return a random batch of size M from training data
        Returned data is in a TensorFlow-friendly format (np array).
        """
        if len(self.epoch_indices) < M:
            if self.discard_leftovers or len(self.epoch_indices) == 0:
                new_epoch_idx, _ = random_subset_indices(self.X_train, len(self.X_train))
                self.epoch_indices = list(new_epoch_idx)
                self.curr_epoch += 1
            else:
                M = len(self.epoch_indices)

        idx = self.epoch_indices[:M]
        self.epoch_indices = self.epoch_indices[M:]

        X_batch = self.X_train[idx, :]
        y_batch = self.y_train[idx, :]
        return X_batch, y_batch

    def reset(self):
        self.epoch_indices = []
        self.curr_epoch = -1

    def normalize_X(self, X):
        return (X - self.X_mean) / self.X_std if self.norm_X else X

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std if self.norm_y else y

    def denormalize_X(self, X):
        return X * self.X_std + self.X_mean if self.norm_X else X

    def denormalize_y(self, y):
        return y * self.y_std + self.y_mean if self.norm_y else y

    def at_end_of_epoch(self, epoch, M):
        if self.discard_leftovers:
            return self.curr_epoch == epoch and len(self.epoch_indices) < M
        return self.curr_epoch == epoch and len(self.epoch_indices) == 0


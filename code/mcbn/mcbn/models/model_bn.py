from copy import deepcopy
import tensorflow as tf

from model import Model
from mcbn.utils.helper import add_to_collection


class ModelBN(Model):
    def __init__(self, n_hidden, K, nonlinearity, bn, do, tau, dataset,
                 in_dim=1, out_dim=1, regression=True, first_layer_do=False):
        self.is_training_ph = tf.placeholder(tf.bool)
        super(ModelBN, self).__init__(n_hidden, K, nonlinearity, bn, do, tau, dataset,
                                      in_dim, out_dim, regression, first_layer_do)

    def run_train_step(self, batch):
        """Train Gamma, Beta, W, b with means and variances from batch"""
        self.train_step.run(feed_dict={self.x: self.dataset.normalize_X(batch[0]),
                                       self.y: self.dataset.normalize_y(batch[1]),
                                       self.is_training_ph: True})

    def predict(self, x):
        """Always maintains previously calculated statistics
        is_training_ph: False since we want to use previous statistics for layer
                        k+1 rather than update from x from layer 1..k
        """
        yHat = self.yHat.eval(feed_dict={self.x: self.dataset.normalize_X(x),
                                         self.is_training_ph: False})
        return self.dataset.denormalize_y(yHat)

    def predict_mc(self, n_samples, x, batch_size):
        """Return mean and var for an MC estimate of x"""
        samples = self.get_mc_samples(n_samples, x, batch_size)
        return self.get_mc_moments(samples)

    def get_mc_samples(self, n_samples, x, batch_size):
        samples = None
        d2 = deepcopy(self.dataset)
        d2.reset()
        for i in range(n_samples):
            batch = d2.next_batch(batch_size)
            d2.reset()
            self.update_layer_statistics(batch[0])
            sample = self.predict(x)
            samples = add_to_collection(sample, samples)
        return samples

    def update_layer_statistics(self, x):
        """Update activation means and variances with the values from x.
        Done by running predictions with is_training_ph True, since this
        updates means and variances (and discarding results)
        """
        self.yHat.eval(feed_dict={self.x: self.dataset.normalize_X(x),
                                  self.is_training_ph: True})

    def classification_accuracy(self, x, y):
        """TEMPORARY. Not for MCBN. Must be run after updating layer statistics."""
        return self.accuracy.eval(feed_dict={self.x: self.dataset.normalize_X(x),
                                             self.y: self.dataset.normalize_y(y),
                                             self.is_training_ph: False})

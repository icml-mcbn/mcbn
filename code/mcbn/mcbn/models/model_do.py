import tensorflow as tf

from model import Model
from mcbn.utils.helper import add_to_collection


class ModelDO(Model):
    def __init__(self, n_hidden, K, nonlinearity, bn, do, tau, dataset,
                 in_dim=1, out_dim=1, regression=True, first_layer_do=False):
        self.keep_prob_ph = tf.placeholder(tf.float32)
        super(ModelDO, self).__init__(n_hidden, K, nonlinearity, bn, do, tau, dataset,
                                      in_dim, out_dim, regression, first_layer_do)

    def run_train_step(self, batch, keep_prob):
        self.train_step.run(feed_dict={self.x: self.dataset.normalize_X(batch[0]),
                                       self.y: self.dataset.normalize_y(batch[1]),
                                       self.keep_prob_ph: keep_prob})

    def predict(self, x, keep_prob):
        yHat = self.yHat.eval(feed_dict={self.x: self.dataset.normalize_X(x),
                                         self.keep_prob_ph: keep_prob})
        return self.dataset.denormalize_y(yHat)

    def predict_mc(self, n_samples, x, keep_prob):
        """Return mean and var for an MC estimate of x"""
        samples = self.get_mc_samples(n_samples, x, keep_prob)
        return self.get_mc_moments(samples)

    def get_mc_samples(self, n_samples, x, keep_prob):
        samples = None
        for i in range(n_samples):
            sample = self.predict(x, keep_prob)
            samples = add_to_collection(sample, samples)
        return samples

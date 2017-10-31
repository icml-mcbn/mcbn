import numpy as np
import tensorflow as tf

from model_components import layer


class Model(object):
    def __init__(self, n_hidden, K, nonlinearity, bn, do, tau, dataset,
                 in_dim=1, out_dim=1, regression=True, first_layer_do=False):
        self.n_hidden = n_hidden
        self.K = K
        self.nonlinearity = nonlinearity
        self.bn = bn
        self.do = do
        self.outputs = []  # per layer
        self.weights = []  # per layer
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.x = tf.placeholder(tf.float32, shape=[None, in_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, out_dim])
        self.regression = regression
        self.first_layer_do = first_layer_do
        self.tau = tau
        self.dataset = dataset

    def initialize(self, l2_lambda, learning_rate, opt_alg='adam'):
        """Builds the model"""

        # Input layer to 1st hidden
        do_in_first = self.do and self.first_layer_do
        o, w = layer(self.x, [self.in_dim, self.K], self, self.nonlinearity, self.bn, do_in_first)
        self.outputs.append(o)
        self.weights.append(w)

        # Subsequent hidden layers
        for _ in range(self.n_hidden - 1):
            o, w = layer(self.outputs[-1], [self.K, self.K], self, self.nonlinearity, self.bn, self.do)
            self.outputs.append(o)
            self.weights.append(w)

        # Output layer
        o, w = layer(self.outputs[-1], [self.K, self.out_dim], self, do=self.do)
        self.outputs.append(o)
        self.weights.append(w)

        # Predictions and cost
        if self.regression:
            self.yHat = self.outputs[-1]
            self.mse_loss = self._get_mse_loss()
            self.l2_weight_regularizer = self._get_weight_regularizer(l2_lambda)
            self.cost = self.mse_loss + self.l2_weight_regularizer
        else:
            self.yHat = tf.nn.softmax(self.outputs[-1])
            self.cross_entropy_loss = self._get_cross_entropy_loss()
            self.cost = self.cross_entropy_loss + self.l2_weight_regularizer

            # Accuracy
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.yHat, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.train_step = self._get_train_step(learning_rate, opt_alg)

    def get_mc_moments(self, samples):
        return np.mean(samples, axis=0), np.var(samples, axis=0) + self.tau ** (-1)

    def _get_weight_regularizer(self, l2_lambda):
        return tf.reduce_sum(
            input_tensor=l2_lambda * tf.stack([tf.nn.l2_loss(w) for w in self.weights])
        )

    def _get_mse_loss(self):
        return tf.losses.mean_squared_error(self.y, self.yHat, weights=0.5)

    def _get_cross_entropy_loss(self):
        return -tf.reduce_sum(self.y * tf.log(self.yHat))

    def _get_train_step(self, learning_rate, opt_alg):
        if opt_alg == 'adam':
            return tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

import tensorflow as tf


def layer(inputs, shape, model, nonlinearity=None, bn=False, do=False):
    """inputs -> Dropout -> FC -> BN -> activation"""
    pos_initial_bias = nonlinearity and nonlinearity == 'relu'

    W = weight_variable(shape)

    # Dropout
    if do:
        fc_inputs = tf.nn.dropout(inputs, model.keep_prob_ph)
    else:
        fc_inputs = inputs

    # FC and BN (no bias if BN)
    if bn:
        fc_outputs = tf.matmul(fc_inputs, W)
        activation_inputs = bn_wrapper(fc_outputs, model)
    else:
        b = bias_variable([shape[1]], pos_initial_bias)
        activation_inputs = tf.matmul(fc_inputs, W) + b

    # Activation
    if nonlinearity is None:
        return activation_inputs, W
    elif nonlinearity == 'sigmoid':
        return tf.nn.sigmoid(activation_inputs), W
    elif nonlinearity == 'relu':
        return tf.nn.relu(activation_inputs), W
    elif nonlinearity == 'tanh':
        return tf.nn.tanh(activation_inputs), W


def bn_wrapper(inputs, model):
    """sliced_inputs extracts the examples from inputs that should be used
    to calculate the BN statistics (means and variances in activation inputs).
    Normally, this is the full input tensor (during training).
    For MCBNDO however, we could append specific examples to be used (batch X or
    training data X), s.t. the statistics are used in the dropped out network
    (hence the same dropout mask is used for setting the statistics and inference
    on the examples not being used to calculate the BN statistics). 
    """
    epsilon = 1e-10

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    def output_train():
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        set_mean = tf.assign(mean, batch_mean)
        set_var = tf.assign(var, batch_var)

        with tf.control_dependencies([set_mean, set_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

    def output_infer():
        # To add printout of batch statistics to stderr:
        # Replace mean, var with tf.Print operations for printing batch statistics
        return tf.nn.batch_normalization(inputs,
                                         mean, # tf.Print(mean, [mean], message="Mean: "),
                                         var, # tf.Print(var, [var], message="Var: "),
                                         beta,
                                         scale,
                                         epsilon)

    return tf.cond(model.is_training_ph, output_train, output_infer)


def weight_variable(shape):
    values = tf.truncated_normal(shape, stddev=0.1)
    # values = np.random.normal(size=tuple(shape)).astype(np.float32)
    return tf.Variable(values)


def bias_variable(shape, pos_initial_bias):
    if pos_initial_bias:
        values = tf.abs(tf.truncated_normal(shape, stddev=0.1))
    else:
        values = tf.truncated_normal(shape, stddev=0.1)
        # values = tf.zeros(shape)
    return tf.Variable(values)

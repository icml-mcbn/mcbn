import time
import os
import sys
import yaml
from functools import partial

import tensorflow as tf
import numpy as np
from numpy import newaxis
from progressbar import ETA, Bar, Percentage, ProgressBar
from skopt import gbrt_minimize

from mnf.wrappers import MNFMC
from mcbn.environment.constants import DATA_PATH
import mcbn.data.dataset_loaders as dl
from mcbn.utils.helper import random_subset_indices, get_logger
from mcbn.utils.metrics import crps, pll, pll_maximum, crps_minimum


# Get evaluation config.
stream_setup = file('setup.yml', 'r')
setup = yaml.load(stream_setup)
stream_setup.close()


# Get results path
RESULTS_PATH = os.path.join(os.getcwd(), 'mnf_results')
TAU_RESULTS_PATH = os.path.join(RESULTS_PATH, 'tau')
PLOT_RESULTS_PATH = os.path.join(RESULTS_PATH, 'plot')


def splitter(dataset_name, seed, test_fraction=0.2):
    # Set random generator seed for reproducible splits
    np.random.seed(seed)

    # Load full dataset
    _, y = dl.load_uci_data_full(dataset_name)

    # Get test examples count
    N = y.shape[0]
    test_count = int(round(test_fraction * N))

    # Get indices of test and training/validation data at random
    test_idx, trainval_idx = random_subset_indices(y, test_count)

    path = os.path.join(DATA_PATH, dataset_name, 'train_cv-test')
    dl.save_indices(path, 'test_indices.txt', test_idx)
    dl.save_indices(path, 'train_cv_indices.txt', trainval_idx)


def get_mc_data(dataset_name, with_validation=True):

    X_train, y_train, X_test, y_test = dl.load_uci_data_test(dataset_name)

    if with_validation:
        val_ind = int(X_train.shape[0] * 8 / 10)
        X_val = X_train[val_ind:, :]
        y_val = y_train[val_ind:]
        X_train = X_train[0:val_ind - 1, :]
        y_train = y_train[0:val_ind - 1]

    # Normalization terms
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    # Set std dev to 1 if feature is constant.
    X_std[np.all(X_train == X_train[0, :], axis=0)] = 1.0

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    if with_validation:
        X_val = (X_val - X_mean) / X_std
        y_val = (y_val - y_mean) / y_std

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std[0], y_mean[0]
    else:
        return (X_train, y_train), (X_test, y_test), y_std[0], y_mean[0]

def run_tau_opt(sess, pyx, x, y_std, y_mean, xvalid, yvalid, find_cu_tau, tau):
    tau = tau[0]

    preds = np.zeros_like(yvalid)
    all_preds = np.zeros([len(yvalid), FLAGS.L])
    widgets = ["Sampling |", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(FLAGS.L, widgets=widgets)
    pbar.start()
    for i in range(FLAGS.L):
        pbar.update(i)
        for j in range(int(xvalid.shape[0] / 100)):
            pyxi = sess.run(pyx, feed_dict={x: xvalid[j * 100:(j + 1) * 100]})
            preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
            all_preds[j * 100:(j + 1) * 100, i] = np.squeeze(pyxi * y_std + y_mean)
        pyxi = sess.run(pyx, feed_dict={x: xvalid[int(xvalid.shape[0] / 100) * 100:]})
        preds[int(xvalid.shape[0] / 100) * 100:] += pyxi / FLAGS.L
        all_preds[int(xvalid.shape[0] / 100) * 100:, i] = np.squeeze(pyxi * y_std + y_mean)
    print()

    samples = all_preds[:, :, newaxis].T.reshape(FLAGS.L, len(all_preds), 1)
    mean, var = np.mean(samples, axis=0), np.var(samples, axis=0) + tau ** (-1)

    if find_cu_tau:
        crps_res = crps(yvalid * y_std + y_mean, mean, tau ** (-1))
        print "CU tau: {}, crps: {}".format(tau, crps_res)
    else:
        crps_res = crps(yvalid * y_std + y_mean, mean, var)
        print "Tau: {}, crps: {}".format(tau, crps_res)

    if find_cu_tau:
        filename = '-tau_cu.txt'
    else:
        filename = '-tau.txt'

    tau_file_path = os.path.join(TAU_RESULTS_PATH, FLAGS.dataset_name + str(FLAGS.dataset_split_seed) + filename)
    fid_tau = open(tau_file_path, 'a')
    if sum(1 for line in open(tau_file_path)) == 0:
        fid_tau.write('tau,crps,CU\n')
    fid_tau.write('%f,%f,%d\n' % (tau, crps_res, find_cu_tau))
    fid_tau.close()

    return crps_res

def train_model(sess, xtrain, ytrain, tf_seed, np_seed, xvalid=False, yvalid=False, n_epochs=False):
    max_overfitting = setup['patience'] * setup['hyperparam_eval_interval']
    best_val_acc = float("inf")
    last_save = 0
    overfitting = 0

    N, dim = xtrain.shape
    iter_per_epoch = int(N / 100)
    input_shape = [None, dim]
    x = tf.placeholder(tf.float32, input_shape, name='x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y_')

    model = MNFMC(N, input_shape=input_shape, flows_q=FLAGS.fq, flows_r=FLAGS.fr, use_z=not FLAGS.no_z,
                  learn_p=FLAGS.learn_p, thres_var=FLAGS.thres_var, flow_dim_h=FLAGS.flow_h)

    tf.set_random_seed(tf_seed)
    np.random.seed(np_seed)
    y = model.predict(x)
    yd = model.predict(x, sample=False)
    pyx = y

    with tf.name_scope('KL_prior'):
        regs = model.get_reg()
        tf.summary.scalar('KL prior', regs)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(y, y_, weights=0.5))
        tf.summary.scalar('Loglike', cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    if FLAGS.anneal:
        number_zero, original_zero = FLAGS.epzero, FLAGS.epochs / 2
        with tf.name_scope('annealing_beta'):
            max_zero_step = number_zero * iter_per_epoch
            original_anneal = original_zero * iter_per_epoch
            beta_t_val = tf.cast((tf.cast(global_step, tf.float32) - max_zero_step) / original_anneal, tf.float32)
            beta_t = tf.maximum(beta_t_val, 0.)
            annealing = tf.minimum(1., tf.cond(global_step < max_zero_step, lambda: tf.zeros((1,))[0], lambda: beta_t))
            tf.summary.scalar('annealing beta', annealing)
    else:
        annealing = 1.

    with tf.name_scope('lower_bound'):
        lowerbound = cross_entropy + annealing * regs
        tf.summary.scalar('Lower bound', lowerbound)

    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(lowerbound, global_step=global_step)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.square(yd - y_))
        tf.summary.scalar('Accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    tf.add_to_collection('logits', y)
    tf.add_to_collection('logits_map', yd)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y_)
    saver = tf.train.Saver(tf.global_variables())

    tf.global_variables_initializer().run()

    idx = np.arange(N)
    steps = 0
    model_dir = './models/mnf_lenet_mnist_fq{}_fr{}_usez{}_thres{}/model/'.format(FLAGS.fq, FLAGS.fr, not FLAGS.no_z,
                                                                                  FLAGS.thres_var)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print('Will save model as: {}'.format(model_dir + 'model'))

    # Train
    if n_epochs:
        max_epochs = n_epochs
    else:
        max_epochs = FLAGS.epochs
    for epoch in range(max_epochs):
        widgets = ["epoch {}/{}|".format(epoch + 1, FLAGS.epochs), Percentage(), Bar(), ETA()]
        pbar = ProgressBar(iter_per_epoch, widgets=widgets)
        pbar.start()
        np.random.shuffle(idx)
        t0 = time.time()
        for j in range(iter_per_epoch):
            steps += 1
            pbar.update(j)
            batch = np.random.choice(idx, 100)
            if j == (iter_per_epoch - 1):
                summary, _ = sess.run([merged, train_step], feed_dict={x: xtrain[batch], y_: ytrain[batch]})
                train_writer.add_summary(summary, steps)
                train_writer.flush()
            else:
                sess.run(train_step, feed_dict={x: xtrain[batch], y_: ytrain[batch]})

        # If using validation data
        if not n_epochs:
            # the accuracy here is calculated by a crude MAP so as to have fast evaluation
            # it is much better if we properly integrate over the parameters by averaging across multiple samples
            tacc = sess.run(accuracy, feed_dict={x: xvalid, y_: yvalid})
            string = 'Epoch {}/{}, valid_acc: {:0.3f}'.format(epoch + 1, FLAGS.epochs, np.sqrt(tacc))
            string += ', dt: {:0.3f}'.format(time.time() - t0)
            print(string)
            sys.stdout.flush()

            if tacc < best_val_acc and epoch > last_save + setup['hyperparam_eval_interval']:
                print('saving best at epoch %d, rmse=%f' % (epoch, np.sqrt(tacc)))
                last_save = epoch
                best_val_acc = tacc
                string += ', model_save: True'
                saver.save(sess, model_dir + 'model')

            if tacc > best_val_acc:
                overfitting += 1
            else:
                overfitting = 0

            if overfitting > max_overfitting:
                break

    if n_epochs:
        last_save = n_epochs
        saver.save(sess, model_dir + 'model')

    return saver, model_dir, pyx, x, last_save


def train():
    (xtrain, ytrain), (xvalid, yvalid), (_, _), y_std, y_mean = get_mc_data(FLAGS.dataset_name)
    min_tau = setup['tau_range'][FLAGS.dataset_name][0]
    max_tau = setup['tau_range'][FLAGS.dataset_name][1]

    # FIND BEST N_EPOCHS; TAU
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            saver, model_dir, pyx, x, n_epochs = train_model(sess, xtrain, ytrain, FLAGS.seed, FLAGS.seed, xvalid, yvalid)

            # CRPS OPTIMIZE TO FIND STD DEV
            print "Finding optimal Tau"
            saver.restore(sess, model_dir + 'model')

            # Make optimization run take extra arguments
            optimize_fun = partial(run_tau_opt, sess, pyx, x, y_std, y_mean, xvalid, yvalid, False)
            tau_opt = gbrt_minimize(optimize_fun,
                                    [(min_tau, max_tau)],
                                    n_random_starts=100,
                                    n_calls=200
                                    )
            opt_tau = tau_opt.x[0]


            # CRPS OPTIMIZE TO FIND STD DEV
            print "Finding optimal CU Tau"
            # Make optimization run take extra arguments
            optimize_fun = partial(run_tau_opt, sess, pyx, x, y_std, y_mean, xvalid, yvalid, True)
            cutau_opt = gbrt_minimize(optimize_fun,
                                    [(min_tau, max_tau)],
                                    n_random_starts=100,
                                    n_calls=200
                                    )
            cu_opt_tau = cutau_opt.x[0]

            print "OPT TAU: {}. CRPS: {}".format(opt_tau, tau_opt.fun)
            print "CU OPT TAU: {}. CRPS: {}".format(cu_opt_tau, cutau_opt.fun)

    tf.reset_default_graph()

    # TRAIN AND EVALUATE FINAL MODEL 5 TIMES WITH DIFFERENT SEED:
    for final_seed in range(FLAGS.seed + 1, FLAGS.seed + 6):
        (xtrain, ytrain), (xtest, ytest), y_std, y_mean = get_mc_data(FLAGS.dataset_name, False)
        with tf.Graph().as_default() as g:
            with tf.Session() as sess:

                # Write csv file column headers if not yet written.
                plot_file_path = os.path.join(PLOT_RESULTS_PATH, FLAGS.dataset_name + '.txt')
                fid = open(plot_file_path, 'a')
                if sum(1 for line in open(plot_file_path)) == 0:
                    fid.write(',MNF const std dev,MNF std dev,y,yHat,run_count,dataset_split_seed\n')

                pll_file_path = os.path.join(RESULTS_PATH, FLAGS.dataset_name + '-pll.txt')
                fid_pll = open(pll_file_path, 'a')
                if sum(1 for line in open(pll_file_path)) == 0:
                    fid_pll.write('dataset_split,run_count,pll result,pll baseline,pll best,pll normalized\n')

                crps_file_path = os.path.join(RESULTS_PATH, FLAGS.dataset_name + '-crps.txt')
                fid_crps = open(crps_file_path, 'a')
                if sum(1 for line in open(crps_file_path)) == 0:
                    fid_crps.write('dataset_split,run_count,crps result,crps baseline,crps best,crps normalized\n')

                rmse_file_path = os.path.join(RESULTS_PATH, FLAGS.dataset_name + '-rmse.txt')
                fid_rmse = open(rmse_file_path, 'a')
                if sum(1 for line in open(rmse_file_path)) == 0:
                    fid_rmse.write('dataset_split,run_count,rmse\n')

                saver, model_dir, pyx, x, _ = train_model(sess, xtrain, ytrain, final_seed, final_seed, xvalid=False, yvalid=False, n_epochs=n_epochs)

                #EVALUATE TEST SET
                preds = np.zeros_like(ytest)
                all_preds = np.zeros([len(ytest), FLAGS.L])
                widgets = ["Sampling |", Percentage(), Bar(), ETA()]
                pbar = ProgressBar(FLAGS.L, widgets=widgets)
                pbar.start()
                for i in range(FLAGS.L):
                    pbar.update(i)
                    for j in range(int(xtest.shape[0] / 100)):
                        pyxi = sess.run(pyx, feed_dict={x: xtest[j * 100:(j + 1) * 100]})
                        preds[j * 100:(j + 1) * 100] += pyxi / FLAGS.L
                        all_preds[j * 100:(j + 1) * 100, i] = np.squeeze(pyxi * y_std + y_mean)
                    pyxi = sess.run(pyx, feed_dict={x: xtest[int(xtest.shape[0] / 100) * 100:]})
                    preds[int(xtest.shape[0] / 100) * 100:] += pyxi / FLAGS.L
                    all_preds[int(xtest.shape[0] / 100) * 100:, i] = np.squeeze(pyxi * y_std + y_mean)

                # FIND PLL AND CRPS
                samples = all_preds[:, :, newaxis].T.reshape(FLAGS.L, len(all_preds), 1)
                mean, var = np.mean(samples, axis=0), np.var(samples, axis=0) + opt_tau ** (-1)
                pll_res = pll(samples, ytest * y_std + y_mean, FLAGS.L, opt_tau)
                crps_res = crps(ytest * y_std + y_mean, mean, var)

                # FIND BASELINE PLL AND CRPS
                pll_baseline = pll(np.array([mean]), ytest * y_std + y_mean, 1, cu_opt_tau)
                crps_baseline = crps(ytest * y_std + y_mean, mean, cu_opt_tau**(-1))

                # FIND OPTIMAL PLL AND CRPS
                pll_best = pll_maximum(mean, ytest * y_std + y_mean)
                crps_best = crps_minimum(mean, ytest * y_std + y_mean)

                # GET NORMALIZED SCORES
                pll_norm = (pll_res - pll_baseline) / (pll_best - pll_baseline)
                crps_norm = (crps_res - crps_baseline) / (crps_best - crps_baseline)

                sample_accuracy = np.sqrt(np.mean((preds-ytest)*(preds-ytest)))
                print('Sample test accuracy: {}'.format(sample_accuracy))

                ytest_u = (ytest * y_std + y_mean)
                preds_u = (preds * y_std + y_mean)
                unnormalized_rmse = np.sqrt(np.mean((preds_u - ytest_u) * (preds_u - ytest_u)))
                print('Sample test accuracy (unnormalized): {}'.format(unnormalized_rmse))

                print('Test uncertainty quality metrics:')
                print "PLL: {}, PLL LOWER: {}, PLL UPPER: {}, NORM: {}".format(pll_res, pll_baseline, pll_best, pll_norm)
                print "CRPS: {}, CRPS LOWER: {}, CRPS UPPER: {}, NORM: {}".format(crps_res, crps_baseline, crps_best, crps_norm)

                all_preds_mean = all_preds.mean(axis=1)
                all_preds_std = all_preds.std(axis=1)

                # Write results to files
                for i in range(len(ytest)):
                    fid.write('%d,%f,%f,%f,%f,%d,%d\n' % (i, np.sqrt(cu_opt_tau**(-1)), all_preds_std[i], ytest_u[i], all_preds_mean[i], final_seed, FLAGS.dataset_split_seed))
                fid_rmse.write('%d,%d,%f\n' % (FLAGS.dataset_split_seed, final_seed, unnormalized_rmse))
                fid_pll.write('%d,%d %f,%f,%f,%f\n' % (FLAGS.dataset_split_seed, final_seed, pll_res, pll_baseline, pll_best, pll_norm))
                fid_crps.write('%d,%d,%f,%f,%f,%f\n' % (FLAGS.dataset_split_seed, final_seed, crps_res, crps_baseline, crps_best, crps_norm))
                fid.close()
                fid_rmse.close()
                fid_pll.close()
                fid_crps.close()

        tf.reset_default_graph()

def main():
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    if not os.path.exists(TAU_RESULTS_PATH):
        os.makedirs(TAU_RESULTS_PATH)

    if not os.path.exists(PLOT_RESULTS_PATH):
        os.makedirs(PLOT_RESULTS_PATH)

    splitter(FLAGS.dataset_name, FLAGS.dataset_split_seed)

    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default=os.path.join(os.getcwd(), 'mnf_logs'),
                        help='Summaries directory')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-epzero', type=int, default=1)
    parser.add_argument('-fq', default=2, type=int)
    parser.add_argument('-fr', default=2, type=int)
    parser.add_argument('-no_z', action='store_true')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-thres_var', type=float, default=0.5)
    parser.add_argument('-flow_h', type=int, default=50)
    parser.add_argument('-L', type=int, default=100)
    parser.add_argument('-anneal', action='store_true')
    parser.add_argument('-learn_p', action='store_true')
    parser.add_argument('-dataset_name', type=str, default='concrete')
    parser.add_argument('-dataset_split_seed', type=int, default=1)
    FLAGS = parser.parse_args()

    main()

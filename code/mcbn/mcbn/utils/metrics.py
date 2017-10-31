import numpy as np
import properscoring as ps
from scipy.misc import logsumexp
from scipy import optimize


def pll(samples, y, T, tau):
    """Return mean Predictive Log Likelihood"""
    # samples shape ex: (950,50), t by col, obs by row
    # y shape ex: (950,1), obs by row

    # Squared l2 norm sums over output dimensions (axis 2)
    sq_l2_norm = np.sum((samples - y) ** 2, axis=2).T

    # We have examples by row, t=1..T by col (axis 1)
    # logsumexp is over t, i.e. axis 1.
    lse = logsumexp(-0.5 * tau * sq_l2_norm, 1)

    # We have examples by row.
    # Reshape lse in case we have a nx1 vector tau (although taking the mean it does not matter)
    pll_per_ex = (lse.reshape(-1,1)
                  - np.log(T)
                  - 0.5 * np.log(2 * np.pi)
                  + 0.5 * np.log(tau))

    return np.mean(pll_per_ex)


def rmse(yHat_array, y_array):
    return np.sqrt(np.sum((yHat_array - y_array) ** 2) / float(len(y_array)))


def crps(y, yHat_means, yHat_variances):
    """yHat_variances is predictive variance, i.e. already includes tau"""
    yHat_std = np.sqrt(yHat_variances)
    return np.mean(ps.crps_gaussian(y, mu=yHat_means, sig=yHat_std))

def crps_minimization(std_dev_array, y, yHat_means):
    """yHat_variances is predictive variance, i.e. already includes tau"""
    return np.mean(ps.crps_gaussian(y, mu=yHat_means, sig=std_dev_array[0]))

def pll_maximum(yHat_2d, y_2d):
    optimal_tau = (yHat_2d - y_2d)**(-2.0)
    return pll(np.array([yHat_2d]), np.array([y_2d]), 1, optimal_tau)

def crps_minimum(yHat_2d, y_2d):
    avg = []
    for i, (yHat_val, y_2d_val) in enumerate(zip(yHat_2d.flatten(), y_2d.flatten())):
        optimal_tau_pll = (yHat_val - y_2d_val) ** (-2.0)
        result = optimize.minimize(crps_minimization,
                    np.sqrt(1.0 / optimal_tau_pll),
                    method='L-BFGS-B',
                    args=(y_2d_val, yHat_val),
                    bounds=[(None, 1e10)],
                    options={'maxiter':100})

        crps_min = crps(y_2d_val, yHat_val, result.x[0]**2.0)
        avg.append(crps_min)

    return np.mean(avg)
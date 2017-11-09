"""
Mutual information experiment.

The mutual information should increase as the correlation between two columns
increases.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from math import log
from scipy.stats import multivariate_normal as mvn
from baxcat.engine import Engine
from baxcat.misc import pflip

sns.set_context('notebook')


def _sample_from_bivariate_discrete(p, n):
    n_rows, n_cols = p.shape
    samples = pflip(p.flatten(), n=n)
    x = np.zeros((n, 2,))
    for s, sample in enumerate(samples):
        i = int(sample / n_cols)
        j = int(sample % n_cols)
        x[s, 0] = i
        x[s, 1] = j

    return x


def _gen_categorical_joint_dist(rho, n_grid):
    span = np.linspace(-3, 3, n_grid)
    p = np.zeros((n_grid, n_grid,))
    cov = np.array([[1, rho], [rho, 1]])
    for i, x in enumerate(span):
        for j, y in enumerate(span):
            dat = np.array([x, y])
            p[i, j] = mvn.pdf(dat, np.zeros(2), cov)

    p /= np.sum(p)

    # entropy/mutual information calculation
    px = np.sum(p, axis=0)
    py = np.sum(p, axis=1)
    hx = -np.sum(px*np.log(px))
    hy = -np.sum(py*np.log(py))
    hxy = -np.sum(p*np.log(p))

    mi = hx + hy - hxy
    return p, mi


def run(n_times=5, n_grid=5, n=200, n_iter=200, vartype='continuous', ax=None):

    rhos = [.1, .25, .4, .5, .75, .9]

    true_mis = np.zeros(len(rhos))
    mis = np.zeros((n_times, len(rhos),))

    for i, rho in enumerate(rhos):
        print('Rho: %1.1f' % (rho,))

        if vartype == 'categorical':
            p, true_mi = _gen_categorical_joint_dist(rho, n_grid)
            metadata = {
                'x_1': {
                    'dtype': 'categorical',
                    'values': [i for i in range(n_grid)]},
                'x_2': {
                    'dtype': 'categorical',
                    'values': [i for i in range(n_grid)]}
                }
        elif vartype == 'continuous':
            true_mi = -.5*log(1.-rho**2.)
            metadata = {}
        else:
            raise ValueError('invalid vartype')

        for t in range(n_times):
            if vartype == 'categorical':
                x = _sample_from_bivariate_discrete(p, n)
            elif vartype == 'continuous':
                sigma = np.array([[1, rho], [rho, 1]])
                mu = np.zeros(2)
                x = np.random.multivariate_normal(mu, sigma, size=n)
            else:
                raise ValueError('invalid vartype')

            df = pd.DataFrame(x, columns=['x_1', 'x_2'])

            engine = Engine(df, n_models=1, metadata=metadata, use_mp=False)
            engine.init_models()
            engine.run(n_iter)

            true_mis[i] = true_mi
            mis[t, i] = engine.mutual_information('x_1', 'x_2', n_samples=500,
                                                  normed=False)

    if ax is not None:
        ax.errorbar(rhos, y=np.mean(mis, axis=0), yerr=np.std(mis, axis=0),
                    label='BaxCat')
        ax.plot(rhos, true_mis, label='True')

        ax.set_xlabel('rho')
        ax.set_ylabel('Mutual Information')
        ax.set_title(vartype)
        ax.legend(loc=0)
    else:
        return mis, true_mis


if __name__ == '__main__':
    plt.figure(tight_layout=True, figsize=(7.5, 3.5,))
    ax = plt.subplot(1, 2, 1)
    run(vartype='categorical', ax=ax)

    ax = plt.subplot(1, 2, 2)
    run(vartype='continuous', ax=ax)

    plt.show()

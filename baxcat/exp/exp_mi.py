"""
Mutual information experiment.

The mutual information should increase as the correlation between two columns
increases.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal as mvn
from baxcat.engine import Engine
from baxcat.misc import pflip

sns.set_context('notebook')


def sample_from_bivariate_discrete(p, n):
    n_rows, n_cols = p.shape
    samples = pflip(p.flatten(), n=n)
    x = np.zeros((n, 2,))
    for s, sample in enumerate(samples):
        i = int(sample / n_cols)
        j = int(sample % n_cols)
        x[s, 0] = i
        x[s, 1] = j

    return x


def categorical_mi():
    n_times = 5
    n_iter = 200
    n_grid = 5
    rhos = [.1, .25, .4, .5, .75, .9]
    n = 200
    m = 1

    mean = np.zeros(2)

    def genp(rho, n_grid):
        span = np.linspace(-3, 3, n_grid)
        p = np.zeros((n_grid, n_grid,))
        cov = np.array([[1, rho], [rho, 1]])
        for i, x in enumerate(span):
            for j, y in enumerate(span):
                dat = np.array([x, y])
                p[i, j] = mvn.pdf(dat, mean, cov)

        p /= np.sum(p)

        # entropy/mutual information calculation
        px = np.sum(p, axis=0)
        py = np.sum(p, axis=1)
        hx = -np.sum(px*np.log(px))
        hy = -np.sum(py*np.log(py))
        hxy = -np.sum(p*np.log(p))

        mi = hx + hy - hxy
        return p, mi

    true_mi = np.zeros(len(rhos))
    mi = np.zeros((n_times, len(rhos),))

    metadata = {
        'x_1': {
            'dtype': 'categorical',
            'values': [i for i in range(n_grid)]},
        'x_2': {
            'dtype': 'categorical',
            'values': [i for i in range(n_grid)]}
        }

    for i, rho in enumerate(rhos):
        p, true_mi[i] = genp(rho, n_grid)
        for t in range(n_times):
            print('Rho: %1.1f, time: %d' % (rho, t+1,))
            x = sample_from_bivariate_discrete(p, n)
            df = pd.DataFrame(x, columns=['x_1', 'x_2'])

            print('\tRunning')
            engine = Engine(df, metadata=metadata, use_mp=False)
            engine.init_models(1)
            engine.run(n_iter, trans_kwargs={'m': m})

            print('\tCalculating MI')
            mi[t, i] = engine.mutual_information('x_1', 'x_2', n_samples=500,
                                                 normed=False)
    rhoary = np.array(rhos)
    true_mi = -.5*np.log(1-rhoary**2.)

    plt.errorbar(rhos, y=np.mean(mi, axis=0), yerr=np.std(mi, axis=0),
                 label='BaxCat')
    plt.plot(rhos, true_mi, label='True')

    plt.xlabel('rho')
    plt.ylabel('Mutual Information')
    plt.legend(loc=0)


def continuous_mi():
    n_times = 10
    n_iter = 200
    n = 200
    m = 1
    rhos = [.1, .25, .4, .5, .75, .9]

    mi = np.zeros((n_times, len(rhos),))

    for i, rho in enumerate(rhos):
        for t in range(n_times):
            print('Rho: %1.1f, time: %d' % (rho, t+1,))
            sigma = np.array([[1, rho], [rho, 1]])
            mu = np.zeros(2)
            x = np.random.multivariate_normal(mu, sigma, size=n)
            df = pd.DataFrame(x, columns=['x_1', 'x_2'])

            print('\tRunning')
            engine = Engine(df, use_mp=False)
            engine.init_models(1)
            engine.run(n_iter, trans_kwargs={'m': m})

            print('\tCalculating MI')
            mi[t, i] = engine.mutual_information('x_1', 'x_2', n_samples=500,
                                                 normed=False)

    rhoary = np.array(rhos)
    true_mi = -.5*np.log(1-rhoary**2.)

    plt.errorbar(rhos, y=np.mean(mi, axis=0), yerr=np.std(mi, axis=0),
                 label='BaxCat')
    plt.plot(rhos, true_mi, label='True')

    plt.xlabel('rho')

if __name__ == '__main__':
    plt.figure(tight_layout=True, figsize=(7.5, 3.5,))
    plt.subplot(1, 2, 1)
    categorical_mi()
    plt.ylabel('Mutual Information')
    plt.title('Categorical')
    plt.legend(loc=0)

    plt.subplot(1, 2, 2)
    continuous_mi()
    plt.title('Continuous')
    plt.legend(loc=0)

    plt.show()

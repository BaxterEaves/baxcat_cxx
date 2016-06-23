import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from baxcat.engine import Engine

from math import sin
from math import cos
from math import pi


sns.set_context('paper')


def _rotate(xy, theta):
    rotmat = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    xy = np.dot(xy, rotmat)
    return xy[:, 0], xy[:, 1]


def gen_ring(n, width=.3):
    j = 0
    x = np.zeros(n)
    y = np.zeros(n)
    while j < n:
        xj = np.random.uniform(-1, 1)
        yj = np.random.uniform(-1, 1)
        h = (xj**2 + yj**2)**.5
        if h > 1.-width and h < 1.:
            x[j] = xj
            y[j] = yj
            j += 1
    return x, y


def gen_diamond(n):
    xy = np.random.uniform(-1, 1, size=(n, 2,))
    x, y = _rotate(xy, pi/4.)
    return x, y


def gen_u(n):
    x = np.random.uniform(-1, 1, size=n)
    y = 2*x**2 + np.random.uniform(-1, 1, size=n)
    return x, y


def gen_wave(n):
    x = np.random.uniform(-1, 1, size=n)
    y = 4 * (x**2 - .5)**2 + np.random.uniform(-1, 1, size=n)/3
    return x, y


def gen_dots(n):
    nper = int(n/4)
    sigma = np.eye(2)*.02
    xy = np.vstack((
        np.random.multivariate_normal([-.6, .6], sigma, size=nper),
        np.random.multivariate_normal([-.6, -.6], sigma, size=nper),
        np.random.multivariate_normal([.6, -.6], sigma, size=nper),
        np.random.multivariate_normal([.6, .6], sigma, size=nper),))
    return xy[:, 0], xy[:, 1]


def onerun(shapefunc, n=250, n_iter=100, n_models=8):
    xo, yo = shapefunc(n)

    s1 = pd.Series(xo)
    s2 = pd.Series(yo)
    df = pd.concat([s1, s2], axis=1)
    df.columns = ['x', 'y']

    # t_list = [b"row_assignment", b"row_alpha", b"column_hypers",
    #           b"column_assignment"]
    engine = Engine(df, n_models=n_models, use_mp=True)
    # engine.run(n_iter, trans_kwargs={'transition_list': t_list})
    engine.run(n_iter)

    xy = engine.sample(['x', 'y'], n=n)
    xe = xy[:, 0]
    ye = xy[:, 1]

    return xo, yo, xe, ye


if __name__ == "__main__":
    n = 1000
    funcs = [gen_ring, gen_diamond, gen_u, gen_wave, gen_dots]
    n_funcs = len(funcs)
    plt.figure(tight_layout=True)
    for i, func in enumerate(funcs):
        xo, yo, xe, ye = onerun(func, n=n, n_iter=200, n_models=8)
        ax = plt.subplot(2, n_funcs, i+1)
        ax.scatter(xo, yo, color='crimson', alpha=.3)

        plt.subplot(2, n_funcs, i+n_funcs+1)
        plt.scatter(xe, ye, color='#333333', alpha=.3)
        plt.xlim(ax.get_xlim())
        plt.ylim(ax.get_ylim())

    plt.show()

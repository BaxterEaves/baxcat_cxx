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

    engine = Engine(df, use_mp=True)
    engine.init_models(n_models)
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

    dfs = []
    for i, func in enumerate(funcs):
        xo, yo, xe, ye = onerun(func, n=n, n_iter=200, n_models=8)

        fname = func.__name__
        s = pd.Series([fname]*n)
        df = pd.concat([s, pd.Series(xo), pd.Series(yo)], axis=1)
        df.columns = ['func', 'x', 'y']
        dfs.append(df)

        ax = plt.subplot(2, n_funcs, i+1)
        ax.scatter(xo, yo, color='crimson', alpha=.3)

        plt.subplot(2, n_funcs, i+n_funcs+1)
        plt.scatter(xe, ye, color='black', alpha=.3)
        plt.xlim(ax.get_xlim())
        plt.ylim(ax.get_ylim())

    plt.show()

    df = pd.concat(dfs, ignore_index=True)
    engine = Engine(df)
    engine.init_models(8)
    engine.run(2000, checkpoint=20)

    engine.convergence_plot()
    plt.show()

    plt.figure(tight_layout=True)

    dfs = []
    for i, func in enumerate(funcs):
        func_name = func.__name__
        x = engine.sample(['x', 'y'], given=[('func', func_name)], n=n)

        ax = plt.subplot(1, n_funcs, i+1)
        plt.title(func.__name__)
        ax.scatter(x[:, 0], x[:, 1], color='black', alpha=.3)
    plt.show()

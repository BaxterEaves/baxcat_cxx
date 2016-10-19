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


def onerun(shapefunc, n=250, n_iter=100, n_models=8, subsample_size=None):
    xo, yo = shapefunc(n)

    s1 = pd.Series(xo)
    s2 = pd.Series(yo)
    df = pd.concat([s1, s2], axis=1)
    df.columns = ['x', 'y']

    engine = Engine(df, n_models=n_models, use_mp=True)
    engine.init_models(subsample_size=subsample_size)
    engine.run(n_iter)

    xy = engine.sample(['x', 'y'], n=n)
    xe = xy[:, 0]
    ye = xy[:, 1]

    return xo, yo, xe, ye


if __name__ == "__main__":
    n = 1000
    subsample_size = .5
    funcs = [gen_ring, gen_diamond, gen_u, gen_wave, gen_dots]
    n_funcs = len(funcs)
    f, axes = plt.subplots(3, n_funcs)
    f.tight_layout()

    dfs = []
    for i, func in enumerate(funcs):
        xo, yo, xe, ye = onerun(func, n=n, n_iter=200, n_models=8,
                                subsample_size=subsample_size)

        fname = func.__name__
        s = pd.Series([fname]*n)
        df = pd.concat([s, pd.Series(xo), pd.Series(yo)], axis=1)
        df.columns = ['func', 'x', 'y']
        dfs.append(df)

        ax = axes[0, i]
        ax.scatter(xo, yo, color='crimson', alpha=.3)

        ax = axes[1, i]
        ax.scatter(xe, ye, color='gray', alpha=.3)
        ax.set_xlim(axes[0, i].get_xlim())
        ax.set_ylim(axes[0, i].get_ylim())

    df = pd.concat(dfs, ignore_index=True)
    engine = Engine(df, n_models=8)
    engine.init_models()
    engine.run(1000, checkpoint=20)

    dfs = []
    for i, func in enumerate(funcs):
        func_name = func.__name__
        x = engine.sample(['x', 'y'], given=[('func', func_name)], n=n)

        ax = axes[2, i]
        ax.scatter(x[:, 0], x[:, 1], color='navy', alpha=.3)
        ax.set_xlim(axes[0, i].get_xlim())
        ax.set_ylim(axes[0, i].get_ylim())
    plt.show()

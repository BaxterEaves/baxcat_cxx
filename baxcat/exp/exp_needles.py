
import random
import itertools as it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from baxcat.engine import Engine

from baxcat.exp.exp_shape import gen_ring

sns.set_context('paper')

mu = np.zeros(2)
cov = np.array([[1, .7], [.7, 1]])

NEEDLE_FUNCS = {
    'correlation': lambda n, kws: np.random.multivariate_normal(mu, cov, n),
    'x': None,
    'ring': lambda n, kws: np.vstack(gen_ring(n, **kws)).T,
    'dots': None,
}

VALID_PAIRTYPES = [k for k in NEEDLE_FUNCS.keys()]


def _gen_data(n_needles, n_distractors, n_rows, pairtype=None, pair_kws=None):
    if pair_kws is None:
        pair_kws = {}

    if pairtype is None:
        pairtype = 'ring'
        pair_kws = {'width': .2}

    needles = []
    col_names = []
    for i in range(n_needles):
        col_names.extend(['n%d_0' % i, 'n%d_1' % i])

        needles.append(NEEDLE_FUNCS[pairtype](n_rows, pair_kws))

    col_names.extend(['d%d' % i for i in range(n_distractors)])
    ndat = np.concatenate(tuple(needles), axis=1)
    data = np.hstack((ndat, np.random.randn(n_rows, n_distractors),))

    assert len(col_names) == data.shape[1]
    df = pd.DataFrame(data, columns=col_names)

    return df


def run(n_models=10, n_iter=200, iter_step=10, n_needles=2, n_distractors=8,
        n_rows=100, pairtype=None, pair_kws=None):

    needle_idxs = [(2*i, 2*i+1,) for i in range(n_needles)]
    needle_cols = list(range(n_needles*2))
    distractor_cols = list(range(n_needles*2, n_needles*2+n_distractors))
    combs = list(it.product(needle_cols, distractor_cols))
    distractor_idxs = random.sample(combs, min(len(combs), 32))

    df = _gen_data(n_needles, n_distractors, n_rows, pairtype, pair_kws)

    engine = Engine(df, n_models=n_models)
    engine.init_models()
    # for model in engine._models:
    #     # XXX: emulates the log grid expected alpha
    #     # e.g. mean(exp(linspace(log(1/n_rows), log(rows))))
    #     # model['state_alpha'] = .5*(n_needles*2. + n_distractors)
    #     model['state_alpha'] = 100.

    # no column_alpha transition
    tlist = [b'row_assignment', b'column_assignment', b'row_alpha',
             b'column_hypers']

    n_steps = int(n_iter/iter_step)
    needle_dps = np.zeros((n_needles, n_steps+1,))
    distractor_dps = np.zeros((len(distractor_idxs), n_steps+1,))
    for i in range(n_steps+1):
        engine.run(iter_step, trans_kwargs={'transition_list': tlist})
        # engine.run(iter_step)

        for nidx, (a, b) in enumerate(needle_idxs):
            a = df.columns[a]
            b = df.columns[b]
            needle_dps[nidx, i] = engine.dependence_probability(a, b)

        for didx, (a, b) in enumerate(distractor_idxs):
            a = df.columns[a]
            b = df.columns[b]
            distractor_dps[didx, i] = engine.dependence_probability(a, b)

    iter_count = np.cumsum([1]+[iter_step]*n_steps)

    for y in distractor_dps:
        plt.plot(iter_count, y, color='gray', alpha=.3)

    for y in needle_dps:
        plt.plot(iter_count, y, color='crimson')

    # plt.gca().set_xscale('log')
    plt.ylim([-.05, 1.05])
    plt.xlim([1, iter_count[-1]])
    plt.show()

    engine.heatmap('dependence_probability')
    plt.show()


if __name__ == "__main__":
    run(n_models=64, iter_step=10, n_rows=100, n_distractors=128, n_iter=500,
        pairtype='correlation')

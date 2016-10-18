import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from baxcat.utils import model_utils as mu

sns.set_context('paper')


MODEL_SCHEMA = {
    'dtypes': [b'continuous'],
    'col_assignment': [0],
    'row_assignments': [[0]],
    'col_suffstats': [[{'n': 0, 'sum_x': None, 'sum_x_sq': None}]],
    'col_hypers': [{'m': 0, 'r': 1, 's': 1, 'nu': 1}]
}

ROW2IDX = [dict((i, i) for i in range(10))]


def impute_conf_ax(means, stds, ns, ax=None):
    models = []
    xs = []
    for mean, std, n in zip(means, stds, ns):
        x = np.random.randn(n)*std + mean
        model = copy.deepcopy(MODEL_SCHEMA)
        model['col_suffstats'][0][0]['n'] = n
        model['col_suffstats'][0][0]['sum_x'] = np.sum(x)
        model['col_suffstats'][0][0]['sum_x_sq'] = np.sum(x**2)

        models.append(model)
        xs.append(x)

        sns.distplot(x, hist=False, ax=ax, norm_hist=False, label='model')

    sns.distplot(np.hstack(xs), hist=False, ax=ax, norm_hist=False,
                 kde_kws={'color': 'crimson', 'lw': 3}, label='combined')

    x, conf = mu.impute(0, 0, ROW2IDX*len(models), models, (-20, 10))
    ax.plot([x, x], [0, ax.get_ylim()[1]], lw=2, ls='--', c='black')
    ax.set_title('Confidence = %f' % conf)


if __name__ == "__main__":
    n = 1000
    f, axes = plt.subplots(4, 2, figsize=(8, 10))

    means = [(0, 0), (-1, 1), (-.25, .25), (0, 0), (-4, 0, 4), (-1, 0, 1),
             (-.1, 0, .1, .05, 4), (-.1, 0, .1, 4.1, 6)]
    stds = [(1, 1), (1, 1), (1, 1), (1, 10), (1, 1, 1), (1, 1, 1),
            (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)]
    ns = [(n, n)]*4 + [(n, n, n)]*2 + [(n, n, n, n, n)]*2

    impute_conf_ax(means[0], stds[0], ns[0], axes[0][0])
    impute_conf_ax(means[1], stds[1], ns[1], axes[1][0])
    impute_conf_ax(means[2], stds[2], ns[2], axes[0][1])
    impute_conf_ax(means[3], stds[3], ns[3], axes[1][1])
    impute_conf_ax(means[4], stds[4], ns[4], axes[2][0])
    impute_conf_ax(means[5], stds[5], ns[5], axes[2][1])
    impute_conf_ax(means[6], stds[6], ns[6], axes[3][0])
    impute_conf_ax(means[7], stds[7], ns[7], axes[3][1])

    plt.savefig('contconf.png', dpi=150)

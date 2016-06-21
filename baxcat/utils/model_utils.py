import numpy as np

from baxcat import dist
from baxcat.misc import pflip

from scipy.misc import logsumexp
from math import log


def _get_view_weights(model, col_idx):
    view_idx = model['column_assignment'][col_idx]
    alpha = model['view_alpha'][view_idx]
    weights = np.array(model['view_counts'][view_idx] + alpha)
    return weights/np.sum(weights)


# private sample utils
# `````````````````````````````````````````````````````````````````````````````
def _sample_component_continuous(model, col_idx, component_idx):
    hypers = model['hyperpriors'][col_idx]
    if component_idx < len(model['col_suffstats']):
        suffstats = model['col_suffstats'][component_idx]
    else:
        suffstats = {'n': 0., 'sum_x': 0., 'sum_x_sq': 0.}

    return dist.nng.sample(suffstats, hypers)


def _sample_component_categorical(model, col_idx, component_idx):
    hypers = model['hyperpriors'][col_idx]
    if component_idx < len(model['col_suffstats']):
        suffstats = model['col_suffstats'][component_idx]
    else:
        k = len(model['col_suffstats'][0]['counts'])
        suffstats = {'n': 0., 'counts': [0.]*k}

    return dist.csd.sample(suffstats, hypers)


DRAWFUNC = {
    'continuous': _sample_component_continuous,
    'categorical': _sample_component_continuous}


def _sample_single_col(model, col_idx, n=1):
    """ Samples data from the column at col_idx """
    weights = _get_view_weights(model, col_idx)
    component_idx = pflip(weights)
    f = DRAWFUNC(model['dtypes'][col_idx])
    x = f(model, col_idx, component_idx)

    return x


def _sample_multi_col(model, col_idxs, n=1):
    n_cols = len(col_idxs)
    assert n_cols > 1

    view_idxs = [model['col_assignment'][vidx] for vidx in range(col_idxs)]

    col2pos = dict((col_idx, i) for i, col_idx in enumerate(col_idxs))

    view2col = dict()
    for col_idx, view_idx in zip(col_idxs, view_idxs):
        view2col[vidx] = view2col.get(view_idx, []) + [col_idx]

    samples = np.zeros((n, n_cols,))
    for i in range(n):
        for view, cols in view2col.items():
            weights = _get_view_weights(model, cols[0])
            component_idx = pflip(weights)
            for col_idx in cols:
                f = DRAWFUNC(model['dtypes'][col_idx])
                x = f(model, col_idx, component_idx)
                samples[i, col2pos[col_idx]] = x


# private probability utils
# `````````````````````````````````````````````````````````````````````````````
def _probability_component_continuous(x, model, col_idx, component_idx):
    hypers = model['hyperpriors'][col_idx]
    if component_idx < len(model['col_suffstats']):
        suffstats = model['col_suffstats'][component_idx]
    else:
        suffstats = {'n': 0., 'sum_x': 0., 'sum_x_sq': 0.}

    return dist.nng.probability(suffstats, hypers)


def _probability_component_categorical(x, model, col_idx, component_idx):
    hypers = model['hyperpriors'][col_idx]
    if component_idx < len(model['col_suffstats']):
        suffstats = model['col_suffstats'][component_idx]
    else:
        k = len(model['col_suffstats'][0]['counts'])
        suffstats = {'n': 0., 'counts': [0.]*k}

    return dist.csd.probability(x, suffstats, hypers)


PROBFUNC = {
    'continuous': _probability_component_continuous,
    'categorical': _probability_component_continuous}


def _probability_single_col(x, model, col_idx):
    """ probability of x from col_idx under model  """
    log_weights = np.log(_get_view_weights(model, col_idx))
    logps = np.zeros(len(log_weights))
    for component_idx, log_weight in enumerate(log_weights):
        f = PROBFUNC(model['dtypes'][col_idx])
        logps[component_idx] = f(x, model, col_idx, component_idx) + log_weight

    return logsumexp(logps)


def _probability_multi_col(x, model, col_idxs):
    raise NotImplementedError()


# Main interface
# `````````````````````````````````````````````````````````````````````````````
def sample(models, col_idxs, n=1):
    # FIXME: docstring
    n_cols = len(col_idxs)
    n_models = len(models)

    # predetermine from which models to sample
    midxs = np.random.randint(n_models, size=n)

    samples = np.zeros((n, n_cols,))
    for i, midx in enumerate(midxs):
        if n_cols == 1:
            x = _sample_single_col(models[midx], col_idxs[0])
            samples[i] = x
        else:
            x = _sample_multi_col(models[midx], col_idxs)
            samples[i, :] = x

    return samples


def probability(x, models, col_idxs):
    assert isinstance(col_idxs, (list, tuple,))
    # FIXME: docstring
    n_cols = len(col_idxs)
    n_models = len(models)

    logps = np.zeros(max(x.shape))
    for i, x_i in enumerate(x):
        logps_m = np.zeros(n_models)
        for j, model in enumerate(models):
            if n_cols == 1:
                lp = _sample_single_col(x_i, model, col_idxs[0])
            else:
                lp = _sample_multi_col(x_i, model, col_idxs)

            logps_m[j] = lp

        logps[i] = logsumexp(logps_m) - log(n_models)

    return logps

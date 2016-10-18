import numpy as np
import itertools as it

from baxcat.dist import nng
from baxcat.dist import csd
from baxcat.misc import pflip

from scipy.misc import logsumexp
from scipy.integrate import quad
from scipy import optimize
from math import log


def _get_hypers_and_suffstats(model, col_idx, row_idx):
    hypers = model['col_hypers'][col_idx]
    view_idx = int(model['col_assignment'][col_idx])
    cat_idx = model['row_assignments'][view_idx][row_idx]
    suffstats = model['col_suffstats'][col_idx][cat_idx]

    return hypers, suffstats


# TODO: make these non-private function that we can unit test
# TODO: make a get_suffstats utility to avoid repeating code
def _probability_component_continuous(x, model, col_idx, component_idx):
    hypers = model['col_hypers'][col_idx]
    if component_idx < len(model['col_suffstats'][col_idx]):
        suffstats = model['col_suffstats'][col_idx][component_idx]
    else:
        suffstats = {'n': 0., 'sum_x': 0., 'sum_x_sq': 0.}

    return nng.probability(x, suffstats, hypers)


def _probability_component_categorical(x, model, col_idx, component_idx):
    hypers = model['col_hypers'][col_idx]
    if component_idx < len(model['col_suffstats'][col_idx]):
        suffstats = model['col_suffstats'][col_idx][component_idx]
    else:
        k = int(model['col_suffstats'][col_idx][0]['k'])
        suffstats = {'n': 0., 'k': k}

    return csd.probability(x, suffstats, hypers)


PROBFUNC = {
    b'continuous': _probability_component_continuous,
    b'categorical': _probability_component_categorical}


def _sample_component_continuous(model, col_idx, component_idx):
    hypers = model['col_hypers'][col_idx]
    if component_idx < len(model['col_suffstats'][col_idx]):
        suffstats = model['col_suffstats'][col_idx][component_idx]
    else:
        suffstats = {'n': 0., 'sum_x': 0., 'sum_x_sq': 0.}

    return nng.sample(suffstats, hypers)


def _sample_component_categorical(model, col_idx, component_idx):
    hypers = model['col_hypers'][col_idx]
    if component_idx < len(model['col_suffstats'][col_idx]):
        suffstats = model['col_suffstats'][col_idx][component_idx]
    else:
        k = int(model['col_suffstats'][col_idx][0]['k'])
        suffstats = {'n': 0., 'k': k}

    return csd.sample(suffstats, hypers)


DRAWFUNC = {
    b'continuous': _sample_component_continuous,
    b'categorical': _sample_component_categorical}


def _get_view_weights(model, col_idx):
    view_idx = int(model['col_assignment'][col_idx])
    alpha = model['view_alphas'][view_idx]
    weights = np.array(model['view_counts'][view_idx] + [alpha])

    return weights/np.sum(weights)


def _get_given_view_weights(model, col_idx, given, return_log=False):
    # XXX: if return_log is True, this is the same as the probability of the
    # givens in same view of col_idx under the model.
    vidx = model['col_assignment'][col_idx]
    vgiven = [(c, x) for c, x in given if model['col_assignment'][c] == vidx]

    if len(vgiven) == 0:
        weights = np.log(_get_view_weights(model, col_idx))
    else:
        weights = np.log(_get_view_weights(model, col_idx))
        for k, (c, x) in it.product(range(len(weights)), vgiven):
            f = PROBFUNC[model['dtypes'][c]]
            weights[k] += f(x, model, c, k)

    # import pdb; pdb.set_trace()
    if return_log:
        return weights-logsumexp(weights)
    else:
        return np.exp(weights-logsumexp(weights))


# private sample utils
# `````````````````````````````````````````````````````````````````````````````
def _sample_single_col(model, col_idx, given=None, n=1):
    """ Samples data from the column at col_idx """
    if given is None:
        weights = _get_view_weights(model, col_idx)
    else:
        weights = _get_given_view_weights(model, col_idx, given)
    component_idx = pflip(weights)
    f = DRAWFUNC[model['dtypes'][col_idx]]
    x = f(model, col_idx, component_idx)

    return x


def _sample_multi_col(model, col_idxs, given=None, n=1):
    n_cols = len(col_idxs)
    assert n_cols > 1

    view_idxs = [model['col_assignment'][col_idx] for col_idx in col_idxs]

    col2pos = dict((col_idx, i) for i, col_idx in enumerate(col_idxs))

    view2col = dict()
    for col_idx, view_idx in zip(col_idxs, view_idxs):
        view2col[view_idx] = view2col.get(view_idx, []) + [col_idx]

    samples = np.zeros((n, n_cols,))
    for i in range(n):
        for view, cols in view2col.items():
            if given is None:
                weights = _get_view_weights(model, cols[0])
            else:
                weights = _get_given_view_weights(model, cols[0], given)
            component_idx = pflip(weights)
            for col_idx in cols:
                f = DRAWFUNC[model['dtypes'][col_idx]]
                x = f(model, col_idx, component_idx)
                samples[i, col2pos[col_idx]] = x

    # import pdb; pdb.set_trace()
    if n > 1:
        return samples
    else:
        return samples[0, :]


# private probability utils
# `````````````````````````````````````````````````````````````````````````````
def _probability_single_col(x, model, col_idx, given=None):
    """ probability of x from col_idx under model  """
    if given is None:
        log_weights = np.log(_get_view_weights(model, col_idx))
    else:
        log_weights = _get_given_view_weights(model, col_idx, given, True)

    logps = np.zeros(len(log_weights))
    for component_idx, log_weight in enumerate(log_weights):
        f = PROBFUNC[model['dtypes'][col_idx]]
        logps[component_idx] = f(x, model, col_idx, component_idx) + log_weight

    return logsumexp(logps)


def _probability_multi_col(x, model, col_idxs, given=None):
    view_idxs = [model['col_assignment'][col_idx] for col_idx in col_idxs]

    col2pos = dict((col_idx, i) for i, col_idx in enumerate(col_idxs))

    view2col = dict()
    for col_idx, view_idx in zip(col_idxs, view_idxs):
        view2col[view_idx] = view2col.get(view_idx, []) + [col_idx]

    view_idxs = list(set(view_idxs))
    clstr_cts = np.ones(len(view_idxs), dtype=int)  # +1 for unonserved cluster
    for i, view_idx in enumerate(view_idxs):
        col_idx = view2col[view_idx][0]
        clstr_cts[i] += len(model['col_suffstats'][col_idx])

    logp = 0.
    for v, view_idx in enumerate(view_idxs):
        cols = view2col[view_idx]
        if given is None:
            log_weights = np.log(_get_view_weights(model, cols[0]))
        else:
            log_weights = _get_given_view_weights(model, cols[0], given, True)
        lp_view = np.copy(log_weights)
        for col_idx in cols:
            y = x[col2pos[col_idx]]
            f = PROBFUNC[model['dtypes'][col_idx]]
            for k, log_weight in enumerate(log_weights):
                lp_view[k] += f(y, model, col_idx, k)
        logp += logsumexp(lp_view)

    return logp


# private continuous confidence utils
# ````````````````````````````````````````````````````````````````````````````
def _nng_t_mean(model, col_idx, row_idx):
    hypers, suffstats = _get_hypers_and_suffstats(model, col_idx, row_idx)
    mn, _, _, _ = nng.update_params(suffstats, hypers)

    return mn


def _continuous_impute_conf(models, row2idx, col_idx, row):
    n_models = len(models)
    if n_models == 1:
        return float('NaN')

    modrow_idxs = get_model_and_row_idxs(row2idx, row)
    means = [_nng_t_mean(models[midx], col_idx, row_idx)
             for midx, row_idx in modrow_idxs]

    a = min(means)
    b = max(means)

    if a == b:
        return 1.

    def f(x):
        return np.exp(surprisal(col_idx, [(row, x)], row2idx, models)[0])

    d, _ = quad(f, a, b)

    return np.exp(-d/2.)


# private categorical confidence utils
# ````````````````````````````````````````````````````````````````````````````
def categorical_pmf(model, col_idx, row_idx):
    hypers, suffstats = _get_hypers_and_suffstats(model, col_idx, row_idx)
    ps = np.exp([csd.probability(i, suffstats, hypers) for i in
                 range(len(suffstats)-2)])

    return ps


def _categorical_impute_conf(models, row2idx, col_idx, row):
    if len(models) == 1:
        return float('NaN')

    modrow_idxs = get_model_and_row_idxs(row2idx, row)
    pmfs = [categorical_pmf(models[midx], col_idx, row_idx)
            for midx, row_idx in modrow_idxs]
    pmf = np.sum(np.array(pmfs), axis=0)/len(models)

    idx = np.argmax(pmf)

    # check if there are entries that are the same
    dups = np.nonzero(pmf == pmf[idx])[0]
    if len(dups) > 1:
        idxs = dups
    else:
        idxs = [idx]

    d = 0.
    for idx in idxs:
        ps = [p[idx] for p in pmfs]
        d += max(ps) - min(ps)

    assert d >= 0. and d <= 1.

    return 1.-d


# `````````````````````````````````````````````````````````````````````````````
def get_model_and_row_idxs(row2idxs, row):
    ixs = []
    for m_ix, cvtr in enumerate(row2idxs):
        ix = cvtr.get(row, None)
        if ix is not None:
            ixs.append((m_ix, ix,))

    return ixs


# Main interface
# `````````````````````````````````````````````````````````````````````````````
def sample(models, col_idxs, given=None, n=1):
    """ Sample from the model(s) likelihood.

    Parameters
    ----------
    models : list(dict)
        List of metadata from and Engine instance (the `models` member).
    col_idxs : list(int)
        List of integer column indices (not columns names).
    n : int
        The number of samples

    Returns
    -------
    x : numpy.ndarray(float)
        Unconverted data array. each row is a sample.
    """
    # FIXME: docstring
    n_cols = len(col_idxs)
    n_models = len(models)

    # predetermine from which models to sample
    midxs = np.random.randint(n_models, size=n)

    samples = np.zeros((n, n_cols,))
    for i, midx in enumerate(midxs):
        if n_cols == 1:
            x = _sample_single_col(models[midx], col_idxs[0], given=given)
            samples[i] = x
        else:
            x = _sample_multi_col(models[midx], col_idxs, given=given)
            samples[i, :] = x

    return samples


# FIXME: Update docstring example
def surprisal(col_idx, queries, row2idxs, models):
    """ The surprisal, or self-information, of a set of observations
,
    Parameters
    ----------
    col_idx : int
        column index
    queries : list(tuple(int, float,))
        list of (row_index, value,) tuples
    row2idxs : list(dict(row index, int))
        Converts the user-facing row indices to model-specific row indices
    models : list
        list of baxcat models

    Returns
    -------
    s : numpy.ndarray(float)
       The surprisal of each observation in queries.

    Example
    -------
    >>> col_idx = 1
    >>> queries = [(0, 1.,), (2, .5,)]
    >>> models = engine.models
    >>> s = surprisal(col_idx, queries, models)
    """

    f = PROBFUNC[models[0]['dtypes'][col_idx]]

    s = np.zeros(len(queries))
    for i, (row, x,) in enumerate(queries):
        modrow_idxs = get_model_and_row_idxs(row2idxs, row)
        n_models = len(modrow_idxs)
        
        assert n_models > 0

        s_row = np.zeros(n_models)
        for j, (midx, row_idx) in enumerate(modrow_idxs):
            try:
                model = models[midx]
            except:
                import pdb; pdb.set_trace()
            assert row_idx is not None

            view_idx = model['col_assignment'][col_idx]
            component_idx = model['row_assignments'][view_idx][row_idx]

            s_row[j] = f(x, model, col_idx, component_idx)

        s[i] = -logsumexp(s_row) + log(n_models)

    return s


def probability(x, models, col_idxs, given=None):
    """ The average probability of x under the models

    Parameters
    ----------
    x : array-like(d,)
        A d-length array where each entry comes from a distrinct feature
    models : list(dict)
        List of metadata from and Engine instance (the `_models` member).
    col_idxs : list(int)
        List of integer column indices (not columns names).

    Returns
    -------
    p : float
        The log probability.

    Example
    -------
    >>> x = [1., 3., 21.1]
    >>> col_idxs = [3, 12, 16]
    >>> model = engine.models
    >>> lp = probability(x, models, col_idxs)
    """

    assert isinstance(col_idxs, (list, tuple,))

    # FIXME: docstring
    n_cols = len(col_idxs)
    n_models = len(models)

    if len(x.shape) == 2:
        n = x.shape[0]
    else:
        n = 1

    logps = np.zeros(n)

    for i in range(n):
        x_i = x[i, :]
        logps_m = np.zeros(n_models)
        for j, model in enumerate(models):
            if n_cols == 1:
                lp = _probability_single_col(x_i[0], model, col_idxs[0],
                                             given=given)
            else:
                lp = _probability_multi_col(x_i, model, col_idxs, given=given)

            logps_m[j] = lp

        logps[i] = logsumexp(logps_m) - log(n_models)

    if n == 1:
        return logps[0]
    else:
        return logps


# FIXME: Update docstring example
def impute(row, col_idx, row2idx, models, bounds):
    """ Impute (choose the max logp value) ad return confidence

    Parameters
    ----------
    row_idx : row index
        The row user-facing index to impute 
    col_idx : int
        The column index to impute
    row2idxs : list(dict(row index, int))
        Converts the user-facing row indices to model-specific row indices
    models : dict
        The baxcat models
    relvals : list(tuple(columns index, value))
        List of column, value tuples that specify the values of all the columns
        dependent with `col_idx` in `row_idx`.
    bounds : list(float) or tuple(float, float)
        A list of values to evaluate (if `col_idx` is categorical) or the lower
        and upper bound for optimization (if `col_idx` is continuous).

    Returns
    -------
    y : float
        The imputed value
    conf : float
        The confidence
    """
    dtype = models[0]['dtypes'][col_idx]
    if dtype == b'categorical':
        queries = [(row, val,) for val in bounds]
        s = surprisal(col_idx, queries, row2idx, models)
        conf = _categorical_impute_conf(models, row2idx, col_idx, row)
        min_idx = np.argmin(s)
        y = queries[min_idx][1]
    else:
        # XXX: Note that fmin function finds the local maxima
        def func(x):
            return surprisal(col_idx, [(row, float(x),)], row2idx, models)
        resbrute = optimize.brute(func, (bounds,), finish=optimize.fmin)
        y = resbrute[0]
        conf = _continuous_impute_conf(models, row2idx, col_idx, row)

    # FIXME: Confidence not implemeted
    return y, conf


def joint_entropy(models, col_idxs, n_samples=1000):
    """ The average joint entropy of the columns under the models.

    Parameters
    ----------
    models : list(dict)
        List of metadata from and Engine instance (the `_models` member).
    col_idxs : list(int)
        List of integer column indices (not columns names).
    n_samples : int
        The number of samples to use for the Monte Carlo approximation.

    Returns
    -------
    h : float
        An estimate of the joint entropy.
    """
    n_models = len(models)

    # predetermine from which models to sample
    midxs = np.random.randint(n_models, size=n_samples)
    logps = np.zeros(n_samples)
    for i, midx in enumerate(midxs):
        x = sample([models[midx]], col_idxs, n=1)
        logps[i] = probability(x, [models[midx]], col_idxs)

    return -np.sum(logps) / n_samples

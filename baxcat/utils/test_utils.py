""" Miscellaneous utilities for inference testing
"""

from scipy.stats import norm
from scipy.stats import dirichlet
from scipy.misc import logsumexp
from baxcat.misc import pflip
from math import log

import pandas as pd
import numpy as np


def _categorical_params(n_cats, sep):
    sep = min(.95, sep)
    # use 5 as an arbitrary cardinality
    return [{'alpha': p} for p in dirichlet.rvs([1.-sep]*5, n_cats)]


def _continuous_params(n_cats, sep):
    return [{'loc': 6.*sep*k, 'scale': 1.} for k in range(n_cats)]


def _categorical_draw(params):
    return pflip(params['p'])


def _continuous_draw(params):
    return norm.rvs(*params)


def _gen_partition(weights, n):
    if isinstance(weights, int):
        weights = [1./weights]*weights
    elif isinstance(weights, (list, np.ndarray,)):
        if abs(sum(weights) - 1.) > 10E12:
            raise ValueError('weight should sum to 1.')
    else:
        msg = "{} is not valid type for weights".format(type(weights))
        raise ValueError(msg)

    k = len(weights)

    if n == 1:
        return [0], weights

    # there should be at least one instance of each of the components in
    # weights
    z = list(range(k))
    if n-k == 1:
        z += [pflip(weights, n=1)]
    elif n-k > 1:
        z += pflip(weights, n=n-k).tolist()

    assert min(z) == 0
    assert max(z) == k-1

    return z, weights


PARAM_FUNCS = {
    'continuous': _continuous_params,
    'categorical': _categorical_params}

LOGPDFS = {
    'categorical': lambda x, alpha: np.log([alpha[xi] for xi in x]),
    'continuous': norm.logpdf}

DRAW = {
    'categorical': lambda alpha: pflip(alpha),
    'continuous': norm.rvs}


# ---
class DataGenerator(object):
    """ Generate and store data and its generating distribution

    Attributes
    ----------
    dtypes : list(str)
        The datatype of each column, either 'continuous' or 'categorical'.
    df : pandas.DataFrame
        The generated data
    params : list(list(dict))
        The distribution parameters the generated the data in each column
    """
    def __init__(self, n_rows, dtypes, view_weights=1, cat_weights=2,
                 cat_sep=.75, seed=1337):
        """
        Parameters
        ----------
        n_rows : int
            The number of rows of data to generate
        dtypes : list(str)
            List of the datatypes for each columns, either 'continuous' or
            'categorical'
        view_weights : int, list(float)
            - int : The number of views. View will have unform weight and be
                represetned at least once.
            - list(float) : The mixture weights of each view. Should sum to 1.
                If len(`view_weights`) = v, then there will at least once
                column in each view.
        cat_weights : int, list(list(float)):
            - int : The number of categories in each view. Each category will
                be represetned at least once.
            - list(list(float)) : A list of the vcategory-weight vectors for
                each view. Each category will be represented at least once in
                each view.
        cat_sep : float, list(float)
            The discriminability (easy of finding clusters) in each column. 0
            results in identical clusters, and 1 results in very different, or
            distant, clusters.
        seed : int
            Random number generator seed
        """
        # FIXME: Get RNG state and restore it once this object is torn down
        self._dtypes = dtypes
        self._n_cols = len(dtypes)
        self._n_rows = n_rows
        self._colpart, self._view_weights = _gen_partition(view_weights,
                                                           self._n_cols)
        n_views = len(self._view_weights)

        if isinstance(cat_sep, (float, np.float32, np.float64,)):
            cat_sep = [cat_sep]*self._n_cols
        elif not isinstance(cat_sep, (list, np.ndarray,)):
            msg = "{} is invalid type for cat_sep".fomat(type(cat_sep))
            raise ValueError(msg)

        if isinstance(cat_weights, int):
            cat_weights = [[1./cat_weights]*cat_weights]*n_views

        self._viewparts = []
        self._cat_weights = []
        for weights in cat_weights:
            prt, wght = _gen_partition(weights, self._n_rows)
            self._viewparts.append(prt)
            self._cat_weights.append(wght)

        self._params = []
        for col, (dtype, sep,) in enumerate(zip(dtypes, cat_sep)):
            k = self._colpart[col]
            n_cats = len(self._cat_weights[k])
            self._params.append(PARAM_FUNCS[dtype](n_cats, sep))

        srs = []
        for col, dtype in enumerate(self._dtypes):
            vidx = self._colpart[col]
            x = []
            for k in self._viewparts[vidx]:
                x.append(DRAW[dtype](**self._params[col][k]))
            srs.append(pd.Series(x))
        self._df = pd.concat(srs, axis=1)

    def log_likelihood(self, x, col):
        """ The ground-truth log likelihood of `x` in column `col`

        Parameters
        ----------
        x : float or int or 1-D list
            data
        col : int
            The column index of `x`

        Returns
        --------
        ll : float
            The loglikelihood
        """
        dtype = self._dtypes[col]
        vidx = self._colpart[col]
        weights = self._cat_weights[vidx]
        n_cats = len(weights)
        n = len(x)
        lls = np.zeros((n, n_cats,))
        for j, (w, params,) in enumerate(zip(weights, self._params[col])):
            lls[:, j] = log(w) + LOGPDFS[dtype](x, **params)

        return logsumexp(lls, axis=1)

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def df(self):
        return self._df

    @property
    def params(self):
        return self._params

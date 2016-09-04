from scipy.stats import norm
from scipy.stats import dirichlet
from baxcat.misc import pflip
from math import log

import pandas as pd
import numpy as np


def _categorical_params(n_cats, sep):
    sep = min(.95, sep)
    # use 5 as an arbitrary cardinality
    return [{'alpha': p} for p in dirichlet.rvs([1.-sep]*5, n_cats)]


def _continuous_params(n_cats, sep):
    return [{'loc': 6.*sep*k, 'scale': 1.,} for k in range(n_cats)]


def _categorical_draw(params):
    return plfip(params['p'])


def _continuous_draw(params):
    return norm.rvs(*params)


def _gen_partition(weights, n):
    if isinstance(weights, int):
        weights = [1./weights]*weights
    elif not isinstance(weights, (list, np.ndarray,)):
        msg = "{} is not valid type for weights".format(type(weights))
        raise ValueError(msg)

    z = pflip(weights, n=n)
    if n == 1:
        z = [z]

    return z, weights


PARAM_FUNCS = {
    'continuous': _continuous_params,
    'categorical': _categorical_params}

LOGPDFS = {
    'categorical': lambda x, p: log(p[x]),
    'continuous': norm.logpdf}

DRAW = {
    'categorical': lambda alpha: pflip(alpha),
    'continuous': norm.rvs}


# ---
class DataGenerator(object):
    def __init__(self, n_rows, dtypes, view_weights=1, cat_weights=3,
                 cat_sep=.75, seed=1337):
        """WRITEME
        """
        self._dtypes = dtypes
        self._n_cols = len(dtypes)
        self._n_rows = n_rows
        self._colpart, self._view_weights = _gen_partition(view_weights,
                                                           self._n_cols)
        n_views = len(self._view_weights)

        if isinstance(cat_sep, (float, np.float32, np.float64,)):
            cat_sep = [cat_sep]*self._n_cols
        elif not isinstance(cat_set, (list, np.ndarray,)):
            msg = "{} is invalid type for cat_sep".fomat(type(cat_setp))
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
        logps = []
        for k, (w, params,) in enumerate(zip(self._cat_weights, self._params)):
            logps.append(log(w) + LOGPDFS[dtype](x, *params))

        return logeumexp(logps)

    @property
    def df(self):
        return self._df

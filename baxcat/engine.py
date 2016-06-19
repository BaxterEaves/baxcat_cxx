
from baxcat.state import BCState
from baxcat.utils import data_utils as du

from multiprocessing.pool import Pool

import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import copy


def _initialize(args):
    data = args[0]
    kwargs = args[1]
    state = BCState(data.T, **kwargs)  # transpose data to col-major
    return state.get_metadata()


def _run(args):
    data = args[0]
    n_iter = args[1]
    n_sec = args[2]
    init_kwargs = args[3]
    trans_kwargs = args[4]

    state = BCState(data.T, **init_kwargs)  # transpose dat to col-major
    t_start = time.time()
    for _ in range(n_iter):
        state.transition(**trans_kwargs)
        t_elapsed = time.time() - t_start
        if t_elapsed > n_sec:
            break

    return state.get_metadata()


class Engine(object):
    """ WRITEME """
    def __init__(self, df, n_models=1, metadata=None, **kwargs):
        """ Initialize """

        guess_n_unique_cutoff = kwargs.get('guess_n_unique_cutoff', 20)

        output = du.process_dataframe(df, metadata, guess_n_unique_cutoff)
        self._data, self._dtypes, self._distargs, self._converters = output

        self._df = df
        self._n_rows, self._n_cols = self._df.shape

        self._row_names = df.index
        self._col_names = df.columns

        self._seed = kwargs.get('seed', None)
        self._n_models = n_models

        if self._seed is not None:
            np.random.seed(self._seed)
            random.seed(self._seed)

        args = []
        for _ in range(self._n_models):
            sd = np.random.randint(2**31-1)
            kwarg = {'dtypes': self._dtypes,
                     'distargs': self._distargs,
                     'seed': sd}
            args.append((self._data, kwarg,))

        self._pool = Pool()
        self._models = self._pool.map(_initialize, args)

    @classmethod
    def load(cls, filename):
        """ Import metadata from a zipped pickle file (pkl.zip) """
        raise NotImplementedError

    @property
    def metadata(self):
        return copy.deepcopy(self._models)

    def save(self, filename):
        """ Export data from a zipped pickle file (.pkl.zip). """
        raise NotImplementedError

    def run(self, n_iter=1, n_sec=float('Inf'), model_idxs=None,
            trans_kwargs=None):
        """ Run the sampler """

        if trans_kwargs is None:
            trans_kwargs = dict()

        if model_idxs is None:
            model_idxs = [i for i in range(self._n_models)]

        args = []
        for idx, model in enumerate(self._models):
            sd = np.random.randint(2**31-1)
            init_kwarg = {'dtypes': self._dtypes,
                          'distargs': self._distargs,
                          'Zv': model['column_assignment'],
                          'Zrcv': model['row_assignments'],
                          'seed': sd}
            args.append((self._data, n_iter, n_sec, init_kwarg, trans_kwargs,))

        self._pool = Pool()
        ud_models = self._pool.map(_run, args)
        for idx, model in zip(model_idxs, ud_models):
            self._models[idx] = model

    def dependence_probability(self, col_a, col_b):
        """ The probabiilty that a dependence exists between a and b. """
        depprob = 0.
        idx_a = self._converters['col2idx'][col_a]
        idx_b = self._converters['col2idx'][col_b]
        for model in self._models:
            asgn_a = model['column_assignment'][idx_a]
            asgn_b = model['column_assignment'][idx_b]
            if asgn_a == asgn_b:
                depprob += 1.
        depprob /= self._n_models

        return depprob

    def mutual_information(self, col_a, col_b, normed=True):
        raise NotImplementedError

    def entropy(self, col):
        # type = self._dtypes[col]
        # if gu.is_continuous_type(type):
        #     # Use simulation to estimate
        #     pass
        # else:
        #     # Enumerate
        #     pass
        raise NotImplementedError

    def joint_entropy(self, cols):
        # if any(gu.is_continuous_type(type) for type in self._dtypes):
        #     # Use simulation to estimate
        #     pass
        # else:
        #     # Enumerate
        #     pass
        raise NotImplementedError

    def conditional_entropy(self, col_a, col_b, n_samples=100):
        """ Conditional entropy, H(A|B), of a given b """
        h_ab = self.joint_entropy([col_a, col_b], n_samples)
        h_b = self.entropy(col_b, n_samples)

        return h_ab - h_b

    def probability(self, x, y=None):
        """ Predictive probability of x_1, ..., x_n given y_1, ..., y_n """
        raise NotImplementedError

    def row_similarity(self, row_a, row_b):
        raise NotImplementedError

    def sample(self, x, y=None, n=1):
        raise NotImplementedError

    def impute(self, row, col):
        raise NotImplementedError

    def get_pandas_df(self):
        raise NotImplementedError

    def convergence_plots(self, model_idxs=None):
        raise NotImplementedError

    def pairwise_func(self, func):
        if func == 'dependence_probability':
            mat = np.eye(self._n_cols)
            for i in range(self._n_cols):
                for j in range(i+1, self._n_cols):
                    col_a = self._converters['idx2col'][i]
                    col_b = self._converters['idx2col'][j]
                    depprob = self.dependence_probability(col_a, col_b)
                    mat[i, j] = depprob
                    mat[j, i] = depprob
            df = pd.DataFrame(mat, index=self._df.columns,
                              columns=self._df.columns)
        elif func == 'mutual_information':
            raise NotImplementedError
        elif func == 'conditional_entropy':
            raise NotImplementedError
        else:
            raise ValueError("%s is an invalid function." % (func,))

        return df

    def heatmap(self, func, **plot_kwargs):
        if func == 'dependence_probability':
            df = self.pairwise_func('dependence_probability')
            plot_kwargs['vmin'] = plot_kwargs.get('vmin', 0.)
            plot_kwargs['vmax'] = plot_kwargs.get('vmax', 1.)
            plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'gray_r')
        elif func == 'mutual_information':
            raise NotImplementedError
        elif func == 'conditional_entropy':
            raise NotImplementedError
        else:
            raise ValueError("%s is an invalid function." % (func,))

        sns.heatmap(df, **plot_kwargs)

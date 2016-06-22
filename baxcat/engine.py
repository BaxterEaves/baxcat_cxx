
from baxcat.state import BCState
from baxcat.utils import data_utils as du
from baxcat.utils import model_utils as mu

from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import copy


def _initialize(args):
    data = args[0]
    kwargs = args[1]

    t_start = time.time()
    state = BCState(data.T, **kwargs)  # transpose data to col-major

    md = state.get_metadata()
    diagnostics = {
        'log_score': state.log_score(),
        'iters': 0,
        'time': time.time() - t_start}

    return md, pd.DataFrame([diagnostics])


def _run(args):
    data = args[0]
    checkpoint = args[1]
    init_kwargs = args[2]
    trans_kwargs = args[3]

    n_iter = trans_kwargs['N']
    if checkpoint is None:
        checkpoint = n_iter
    else:
        trans_kwargs['N'] = checkpoint

    diagnostics = []
    state = BCState(data.T, **init_kwargs)  # transpose dat to col-major
    for i in range(int(n_iter/checkpoint)):
        t_start = time.time()
        state.transition(**trans_kwargs)
        t_iter = time.time() - t_start

        diagnostic = {
            'log_score': state.log_score(),
            'iters': checkpoint,
            'time': t_iter}
        diagnostics.append(diagnostic)

    md = state.get_metadata()
    return md, pd.DataFrame(diagnostics)


class Engine(object):
    """ WRITEME """
    def __init__(self, df, n_models=1, metadata=None, **kwargs):
        """ Initialize """

        guess_n_unique_cutoff = kwargs.get('guess_n_unique_cutoff', 20)
        use_mp = kwargs.get('use_mp', True)

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

        if use_mp:
            self._pool = Pool()
            self._mapper = self._pool.map
        else:
            self._mapper = lambda func, args: [func(arg) for arg in args]

        self._models = []
        self._diagnostic_tables = []

        res = self._mapper(_initialize, args)
        for model, diagnostics in res:
            self._models.append(model)
            self._diagnostic_tables.append(diagnostics)

    @classmethod
    def load(cls, filename):
        """ Import metadata from a zipped pickle file (pkl.zip) """
        raise NotImplementedError

    @ property
    def column_info(self):
        s_dtypes = pd.Series(self._dtypes, index=self._col_names)
        s_distargs = pd.Series([a[0] for a in self._distargs],
                               index=self._col_names)

        cols = ['dtype', 'cardinality']

        df = pd.concat([s_dtypes, s_distargs], axis=1)
        df.columns = cols

        df['cardinality'][df['dtype'] != 'categorical'] = None
        return df

    @property
    def metadata(self):
        return copy.deepcopy(self._models)

    def save(self, filename):
        """ Export data from a zipped pickle file (.pkl.zip). """
        raise NotImplementedError

    def run(self, n_iter=1, checkpoint=None, model_idxs=None,
            trans_kwargs=None):
        """ Run the sampler """

        if trans_kwargs is None:
            trans_kwargs = dict()

        trans_kwargs['N'] = n_iter

        if model_idxs is None:
            model_idxs = [i for i in range(self._n_models)]

        args = []
        for idx, model in enumerate(self._models):
            sd = np.random.randint(2**31-1)
            init_kwarg = {'dtypes': self._dtypes,
                          'distargs': self._distargs,
                          'Zv': model['col_assignment'],
                          'Zrcv': model['row_assignments'],
                          'col_hypers': model['col_hypers'],
                          'seed': sd}
            args.append((self._data, checkpoint, init_kwarg, trans_kwargs,))

        res = self._mapper(_run, args)
        for idx, (model, diagnostics) in zip(model_idxs, res):
            self._models[idx] = model
            diag_i = self._diagnostic_tables[idx]
            self._diagnostic_tables[idx] = diag_i.append(diagnostics)

    def dependence_probability(self, col_a, col_b):
        """ The probabiilty that a dependence exists between a and b. """
        depprob = 0.
        idx_a = self._converters['col2idx'][col_a]
        idx_b = self._converters['col2idx'][col_b]
        for model in self._models:
            asgn_a = model['col_assignment'][idx_a]
            asgn_b = model['col_assignment'][idx_b]
            if asgn_a == asgn_b:
                depprob += 1.
        depprob /= self._n_models

        return depprob

    def mutual_information(self, col_a, col_b, normed=True, n_samples=1000):
        """ The mutual information, I(A, B), between two columns.

        Parameters
        ----------
        col_a : indexer
            The name of the first column
        col_b : indexer
            The name of the second column
        normed : bool
            If True, the mutual information, I, is normed according to the
            symmertic uncertainty, U = 2*I(A, B)/(H(A) + H(B)).
        n_samples : int
            The number of samples to use for the Monte Carlo approximation
            (if nored if `col` is categorical).

        Returns
        -------
        The mutual information between `col_a` and `col_b`.
        """

        idx_a = self._converters['col2idx'][col_a]
        idx_b = self._converters['col2idx'][col_b]

        models = []
        for model in self._models:
            if model['col_assignment'][idx_a] == model['col_assignment'][idx_b]:
                models.append(model)

        if len(models) == 0:
            return 0.0
        else:
            h_a = self.entropy(col_a, n_samples=n_samples)
            h_b = self.entropy(col_b, n_samples=n_samples)
            h_ab = mu.joint_entropy(models, [idx_a, idx_b], n_samples)
            mi = h_a + h_b - h_ab
            if normed:
                # normalize using symmetric uncertainty
                mi = 2.*mi/(h_a + h_b)

            return mi

    def entropy(self, col, n_samples=500):
        """ The entropy of a column.

        Parameters
        ----------
        col : indexer
            The name of the column
        n_samples : int
            The number of samples to use for the Monte Carlo approximation
            (if nored if `col` is categorical).

        Returns
        -------
        h : float
            The entropy of `col`.
        """

        col_idx = self._converters['col2idx'][col]
        dtype = self._dtypes[col_idx]

        # Unless x is enumerable (is categorical), we approximate h(x) using
        # an importance sampling extimate of h(x) using p(x) as the importance
        # function.
        if dtype == 'categorical':
            k = self._distargs[col_idx]
            for i, x in enumerate(range(k)):
                logps = mu.probability(x, self._models, (col_idx,))

            h = -np.sum(logps) / n_samples
        else:
            x = mu.sample(self._models, (col_idx,))
            logps = mu.probability(x, self._models, (col_idx,))

            h = -np.sum(np.exp(logps)*logps)

        return h

    def conditional_entropy(self, col_a, col_b, n_samples=1000):
        """ Conditional entropy, H(A|B), of a given b """
        col_idxs = [self._converters['col2idx'][col_a],
                    self._converters['col2idx'][col_b]]
        h_ab = mu.joint_entropy(self._models, col_idxs, n_samples)
        h_b = self.entropy(col_b, n_samples)

        return h_ab - h_b

    def probability(self, x, cols, y=None):
        """ Predictive probability of x_1, ..., x_n given y_1, ..., y_n """
        col_idxs = [self._converters['col2idx'][col] for col in cols]
        return mu.probability(x, self._models, col_idxs)

    def row_similarity(self, row_a, row_b):
        raise NotImplementedError

    def sample(self, cols, n=1):
        col_idxs = [self._converters['col2idx'][col] for col in cols]
        return mu.sample(self._models, col_idxs, n=n)

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

        g = sns.clustermap(df, **plot_kwargs)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

    def convergence_plot(self, ax=None):
        if ax is None:
            ax = plt.gca()

        for table in self._diagnostic_tables:
            x = np.cumsum(table['time'].values)
            y = table['log_score'].values

            ax.plot(x, y)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('log score')

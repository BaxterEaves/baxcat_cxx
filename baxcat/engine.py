
from baxcat.state import BCState
from baxcat.utils import data_utils as du
from baxcat.utils import model_utils as mu
from baxcat.utils import plot_utils as pu

from math import exp
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import copy

sns.set_style("white")

# cpickle is faster, but not everybody has it.
try:
    import cpickle as pkl
except ImportError:
    import pickle as pkl


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

    return md, [diagnostics]


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
    return md, diagnostics


# -----------------------------------------------------------------------------
class Engine(object):
    """ WRITEME """
    def __init__(self, df=None, metadata=None, **kwargs):
        """ Initialize """

        if df is None:
            raise ValueError('Give me some data (;-_-)')

        self._init_args = {'df': df, 'metadata': metadata, 'kwargs': kwargs}

        guess_n_unique_cutoff = kwargs.get('guess_n_unique_cutoff', 20)
        use_mp = kwargs.get('use_mp', True)

        output = du.process_dataframe(df, metadata, guess_n_unique_cutoff)
        self._data, self._dtypes, self._distargs, self._converters = output

        self._df = df
        self._n_rows, self._n_cols = self._df.shape
        self._metadata = metadata

        self._row_names = df.index
        self._col_names = df.columns

        self._seed = kwargs.get('seed', None)

        if self._seed is not None:
            np.random.seed(self._seed)
            random.seed(self._seed)

        if use_mp:
            self._pool = Pool()
            self._mapper = self._pool.map
        else:
            self._pool = None
            self._mapper = lambda func, args: [func(arg) for arg in args]

        self._models = []
        self._diagnostic_tables = []

    def init_models(self, n_models):
        """ Intialize a number of cross-categorization models.

        Parameters
        ----------
        n_models : int
            The number of models to initialize.
        """
        if len(self._models) != 0:
            raise NotImplementedError('Cannot add more models.')

        self._n_models = n_models
        args = []
        for _ in range(self._n_models):
            sd = np.random.randint(2**31-1)
            kwarg = {'dtypes': self._dtypes,
                     'distargs': self._distargs,
                     'seed': sd}
            args.append((self._data, kwarg,))

        res = self._mapper(_initialize, args)
        for model, diagnostics in res:
            self._models.append(model)
            self._diagnostic_tables.append(diagnostics)

    @classmethod
    def load(cls, filename):
        """ Create an engine given metadata from a pickle file.

        Parameters
        ----------
        filename : str
        """
        with open(filename, 'rb') as f:
            dat = pkl.load(f)
            self = cls(**dat['init_args'])
            for key, val in dat['cls_attrs'].items():
                setattr(self, '_' + key, val)

            random.setstate(dat['rng_state']['py'])
            np.random.set_state(dat['rng_state']['np'])

        return self

    def save(self, filename):
        """ Save to a zipped pickle file.

        The resulting file can be used to initialize Engine object using
        `Engine.load(filename)`.

        Parameters
        ----------
        filename : str
        """
        dat = {
            'init_args': self._init_args,
            'rng_state': {
                'np': np.random.get_state(),
                'py': random.getstate()},
            'cls_attrs': {
                'models': self._models,
                'n_models': self._n_models,
                'diagnostic_tables': self._diagnostic_tables}}

        with open(filename, 'wb') as f:
            pkl.dump(dat, f)

    @property
    def models(self):
        return copy.deepcopy(self._models)

    def diagnostics(self, model_idxs=None):
        if model_idxs is None:
            model_idxs = [i for i in range(self._n_models)]
        return [pd.DataFrame(self._diagnostic_tables[m]) for m in model_idxs]

    def col_info(self):
        """ Get a DataFrame with basic info about the columns """
        s_dtypes = pd.Series(self._dtypes, index=self._col_names)
        s_distargs = pd.Series([a[0] for a in self._distargs],
                               index=self._col_names)

        cols = ['dtype', 'cardinality']

        df = pd.concat([s_dtypes, s_distargs], axis=1)
        df.columns = cols

        df['cardinality'][df['dtype'] != 'categorical'] = None
        return df

    def run(self, n_iter=1, checkpoint=None, model_idxs=None,
            trans_kwargs=None):
        """ Run the sampler.

        Parameters
        ----------
        n_iter : int
            The number of iterations to run the sampler.
        checkpoint : int
            Collect diagnostic data every `checkpoint` iterations. By default,
            collects only at the end of the run.
        model_idxs : list(int)
            The indices of the models to run, if not all models.
        trans_kwargs : dict
            Keyword arguments sent to `BCState.transition`
        """

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
                          'state_alpha': model['state_alpha'],
                          'view_alphas': model['view_alphas'],
                          'seed': sd}
            args.append((self._data, checkpoint, init_kwarg, trans_kwargs,))

        res = self._mapper(_run, args)
        for idx, (model, diagnostics) in zip(model_idxs, res):
            self._models[idx] = model
            self._diagnostic_tables[idx].extend(diagnostics)

    def sample(self, cols, given=None, n=1):
        """ Draw samples from cols """
        # TODO: make sure that given goes not caontain columns rom cols
        if given is not None:
            given = du.convert_given(given, self._dtypes, self._converters)

        col_idxs = [self._converters['col2idx'][col] for col in cols]

        data_out = mu.sample(self._models, col_idxs, given=given, n=n)

        x = du.convert_data(data_out, cols, self._dtypes, self._converters,
                            to_val=True)

        if x.shape == (1, 1,):
            return x[0, 0]
        elif x.shape[0] == 1:
            return x[0, :]
        else:
            return x

    def probability(self, x, cols, given=None):
        """ Predictive probability of x_1, ..., x_n given y_1, ..., y_n

        Parameters
        ----------
        x : numpy.ndarray(2,)
            2-D numpy array where each row is a set of observations and each
            column corresponds to a feature.
        cols : list
            The names of each column/feature of `x`.
        given : list(tuple)
            List of (name, value,) conditional contraints for the probability

        Returns
        -------
        logps : numpy.ndarray
        """
        # TODO: make sure that given goes not caontain columns rom cols
        x = du.format_query_data(x)
        col_idxs = [self._converters['col2idx'][col] for col in cols]
        x_cnv = du.convert_data(x, cols, self._dtypes, self._converters)

        if given is not None:
            given = du.convert_given(given, self._dtypes, self._converters)

        return mu.probability(x_cnv, self._models, col_idxs)

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

    def row_similarity(self, row_a, row_b):
        """ The similarity between two rows  in terms of their partitions. """
        if row_a == row_b:
            # XXX: we will assume that the user meant to do this
            return 1.0

        idx_a = self._converters['row2idx'][row_a]
        idx_b = self._converters['row2idx'][row_b]

        sim = np.zeros(self._n_models)
        for midx, model in enumerate(self._models):
            n_views = len(model['row_assignments'])
            sim[midx] = sum(asgn[idx_a] == asgn[idx_b] for asgn in
                            model['row_assignments'])
            sim[midx] /= float(n_views)
        return np.mean(sim)

    def impute(self, row, col, min_conf=0.):
        # XXX: Holding off on this because it's not entirely obvious how to
        # implement an intuitive, and mathematically reasonable, confidence
        # measure for multimodal continuous data.
        raise NotImplementedError

    # TODO: allow multiple columns for joint entropy
    def entropy(self, col, n_samples=500):
        """ The entropy of a column.

        Notes
        -----
        Returns differential entropy for continuous feature (obviously).

        Parameters
        ----------
        col : indexer
            The name of the column
        n_samples : int
            The number of samples to use for the Monte Carlo approximation
            (if `col` is categorical).

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
            k = self._distargs[col_idx][0]
            x = np.array([[i] for i in range(k)])
            logps = mu.probability(x, self._models, (col_idx,))
            assert logps.shape == (k,)
            h = -np.sum(np.exp(logps)*logps)
        else:
            x = mu.sample(self._models, (col_idx,), n=n_samples)
            logps = mu.probability(x, self._models, (col_idx,))

            h = -np.sum(logps) / n_samples

        return h

    def mutual_information(self, col_a, col_b, normed=True, linfoot=False,
                           n_samples=1000):
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
        mi : float
            The mutual information between `col_a` and `col_b`.
        """

        if linfoot:
            normed = False

        idx_a = self._converters['col2idx'][col_a]
        idx_b = self._converters['col2idx'][col_b]

        models = []
        for model in self._models:
            if model['col_assignment'][idx_a] == model['col_assignment'][idx_b]:
                models.append(model)

        if len(models) == 0:
            mi = 0.0
        else:
            h_a = self.entropy(col_a, n_samples=n_samples)
            h_b = self.entropy(col_b, n_samples=n_samples)
            h_ab = mu.joint_entropy(models, [idx_a, idx_b], n_samples)
            mi = h_a + h_b - h_ab

            # XXX: Differential entropy can be negative. Here we prevent
            # negative mutual information.
            mi = max(mi, 0.)
            if normed:
                # normalize using symmetric uncertainty
                mi = 2.*mi/(h_a + h_b)

        if linfoot:
            mi = (1. - exp(-2*mi))**.5

        return mi

    def conditional_entropy(self, col_a, col_b, n_samples=1000):
        """ Conditional entropy, H(A|B), of a given b.

        Implementation notes
        --------------------
        Uses MonteCarlo integration at least in the joint entropy component.

        Parameters
        ----------
        col_a : indexer
            The name of the first column
        col_b : indexer
            The name of the second column
        n_samples : int
            The number of samples to use for the Monte Carlo approximation
            (if nored if `col` is categorical).

        Returns
        -------
        h_c : float
            The conditional entropy of `col_a` given `col_b`.
        """
        col_idxs = [self._converters['col2idx'][col_a],
                    self._converters['col2idx'][col_b]]
        h_ab = mu.joint_entropy(self._models, col_idxs, n_samples)
        h_b = self.entropy(col_b, n_samples)
        h_c = h_ab - h_b

        return h_c

    # TODO: offload to external fuction that can be parallelized
    def pairwise_func(self, func, n_samples=500):
        """ Do a function over all paris of columns/rows """
        mat = np.eye(self._n_cols)
        if func == 'dependence_probability':
            for i in range(self._n_cols):
                for j in range(i+1, self._n_cols):
                    col_a = self._converters['idx2col'][i]
                    col_b = self._converters['idx2col'][j]
                    depprob = self.dependence_probability(col_a, col_b)
                    mat[i, j] = depprob
                    mat[j, i] = depprob
        elif func in ['mutual_information', 'linfoot']:
            if func == 'linfoot':
                linfoot = True
            else:
                linfoot = False

            for i in range(self._n_cols):
                for j in range(i+1, self._n_cols):
                    col_a = self._converters['idx2col'][i]
                    col_b = self._converters['idx2col'][j]
                    if i == j:
                        mi = 1.
                    else:
                        mi = self.mutual_information(
                            col_a, col_b, linfoot=linfoot, n_samples=n_samples)
                    mat[i, j] = mi
                    mat[j, i] = mi
        elif func == 'conditional_entropy':
            mat = np.eye(self._n_cols)
            for i in range(self._n_cols):
                for j in range(self._n_cols):
                    col_a = self._converters['idx2col'][i]
                    col_b = self._converters['idx2col'][j]
                    if i == j:
                        h = self.entropy(col_a, n_samples)
                    else:
                        h = self.conditional_entropy(col_a, col_b, n_samples)

                    mat[i, j] = h
        else:
            raise ValueError("%s is an invalid function." % (func,))

        df = pd.DataFrame(mat, index=self._df.columns, columns=self._df.columns)

        return df

    def heatmap(self, func, n_samples=100, plot_kwargs=None):
        """ Heatmap of a pairwise function """
        if plot_kwargs is None:
            plot_kwargs = {}

        if func == 'dependence_probability':
            plot_kwargs['vmin'] = plot_kwargs.get('vmin', 0.)
            plot_kwargs['vmax'] = plot_kwargs.get('vmax', 1.)
            plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'gray_r')
        elif func in ['mutual_information', 'linfoot']:
            plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'gray_r')

        df = self.pairwise_func(func, n_samples=n_samples)

        g = sns.clustermap(df, **plot_kwargs)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

    def convergence_plot(self, ax=None, log_x_axis=True, min_time=0.):
        """ Plot the log score of each model as a function of time. """
        if ax is None:
                ax = plt.gca()

        xs = []
        for table in self.diagnostics():
            x = np.cumsum(table['time'].values)
            t = np.nonzero(x > min_time)[0]
            if len(t) > 0:
                t = t[0]
                y = table['log_score'].values
                xs.extend([x[t], x[-1]])
                ax.plot(x[t:], y[t:])
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('log score')

        if log_x_axis:
            ax.set_xscale('log')
            ax.set_xlim([min(xs), max(xs)])

    def plot_state(self, state_idx):
        # raise NotImplementedError()
        model = self._models[state_idx]
        init_kwargs = {'dtypes': self._dtypes,
                       'distargs': self._distargs,
                       'Zv': model['col_assignment'],
                       'Zrcv': model['row_assignments'],
                       'col_hypers': model['col_hypers'],
                       'state_alpha': model['state_alpha'],
                       'view_alphas': model['view_alphas']}
        state = BCState(self._data.T, **init_kwargs)
        model_logps = state.get_logps()
        pu.plot_cc_model(self._data, model, model_logps, self._df.index,
                         self._df.columns)

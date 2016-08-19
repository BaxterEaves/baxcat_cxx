""" The main interface to `baxcat_cxx`
"""

import random
import time
import copy
from math import exp
from multiprocessing.pool import Pool

from baxcat.state import BCState
from baxcat.utils import data_utils as du
from baxcat.utils import model_utils as mu
from baxcat.utils import plot_utils as pu
from baxcat import metrics

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

    metadata = state.get_metadata()
    diagnostics = {
        'log_score': state.log_score(),
        'iters': 0,
        'time': time.time() - t_start}

    return metadata, [diagnostics]


def _run(args):
    data = args[0]
    checkpoint = args[1]
    t_id = args[2]
    quiet = args[3]
    init_kwargs = args[4]
    trans_kwargs = args[5]

    # create copy of trans_kwargs so we don't mutate
    trans_kwargs = dict(trans_kwargs)
    n_iter = trans_kwargs['N']
    if checkpoint is None:
        checkpoint = n_iter
        n_sweeps = 1
    else:
        trans_kwargs['N'] = checkpoint
        n_sweeps = int(n_iter/checkpoint)

    diagnostics = []
    state = BCState(data.T, **init_kwargs)  # transpose dat to col-major
    for i in range(n_sweeps):
        t_start = time.time()
        state.transition(**trans_kwargs)
        t_iter = time.time() - t_start

        n_views = state.n_views
        log_score = state.log_score()

        diagnostic = {
            'log_score': log_score,
            'n_views': n_views,
            'iters': checkpoint,
            'time': t_iter}
        diagnostics.append(diagnostic)

        if not quiet:
            msg = "Model {}:\n\t+ sweep {} of {} in {} sec."
            msg += "\n\t+ log score: {}"
            msg += "\n\t+ n_views: {}\n"

            print(msg.format(t_id, i, n_sweeps, t_iter, log_score, n_views))

    metadata = state.get_metadata()

    return metadata, diagnostics


# ---
class Engine(object):
    """
    The inference engine interface.

    Attributes
    ----------
    columns : list(index)
        A list of columns names
    models : metadata
        A list of metadata objects that store all the relvant information for
        a set of cross-categorization states.
    """

    def __init__(self, df=None, metadata=None, **kwargs):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            The data for inference.
        metadata : dict
            Column metadata to speed processing. Providing more data in
            `metadata` speeds up the processing of `df` by obviating the need
            to infer data types and create value maps.
        seed : integer
            Positive integer seed for the random number generators.
        use_mp : bool, optional
            If True (default), model-parallel tasts are run in parallel.

        Example
        -------
        Initialize with partial metadata
        >>> data = pd.read_csv('example/zoo.csv', index_col=0)
        >>> metadata = {
        ...     'stripes': {
        ...         'dtype': 'categorical',
        ...         'values': [0, 1]
        ...     }
        ... }
        >>> engine = Engine(df, metadata)

        """

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
            self._mapper = lambda func, args: list(map(func, args))

        self._models = []
        self._n_models = 0
        self._diagnostic_tables = []

    def init_models(self, n_models, structureless=False):
        """ Intialize a number of cross-categorization models.

        Parameters
        ----------
        n_models : int
            The number of models to initialize.

        Other parameters
        ----------------
        structureless : bool, optional
            If True, initalize each cross-categorization state to have one
            view and one category (default is False). This is primiarily for
            debugging and visualization, and negatively affects inference.
        """
        if self._n_models != 0:
            raise NotImplementedError('Cannot add more models.')

        self._n_models = n_models
        args = []
        for _ in range(self._n_models):
            sd = np.random.randint(2**31-1)
            kwarg = {'dtypes': self._dtypes,
                     'distargs': self._distargs,
                     'seed': sd}
            if structureless:
                kwarg['Zv'] = [0]*self._n_cols
                kwarg['Zrcv'] = [[0]*self._n_rows]

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
    def columns(self):
        return self._col_names

    @property
    def models(self):
        return copy.deepcopy(self._models)

    def diagnostics(self, model_idxs=None):
        """ Get diagnostics for each model.

        There will be no diagnostics if `checkpoint` is not specified in
        `Engine.run`.

        Parameters
        ----------
        model_idxs : list(int)
            List of model integer indices for which to collect diagnostics.

        Returns
        -------
        list(pandas.DataFrame)
            A diagnostics table for each model. Each table has columns
            `log_score` (log score of the model), `iters` (number of inferece
            iterations since the last checkpoint), and `time` (the number of
            seconds ellapsed since the last checkpoint).

        Example
        -------
        >>> df = pd.read_csv('examples/zoo.csv')
        >>> engine = Engine(df, seed=1337)
        >>> engine.init_models(2)
        >>> engine.run(10, checpoint=2)
        >>> engine.diagnostics([1])
        """
        if model_idxs is None:
            model_idxs = list(range(self._n_models))
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
            trans_kwargs=None, quiet=True):
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
            model_idxs = list(range(self._n_models))

        args = []
        for idx in model_idxs:
            model = self._models[idx]
            sd = np.random.randint(2**31-1)
            init_kwarg = {'dtypes': self._dtypes,
                          'distargs': self._distargs,
                          'Zv': model['col_assignment'],
                          'Zrcv': model['row_assignments'],
                          'col_hypers': model['col_hypers'],
                          'state_alpha': model['state_alpha'],
                          'view_alphas': model['view_alphas'],
                          'seed': sd}
            args.append((self._data, checkpoint, idx, quiet, init_kwarg,
                         trans_kwargs,))

        res = self._mapper(_run, args)
        for idx, (model, diagnostics) in zip(model_idxs, res):
            self._models[idx] = model
            self._diagnostic_tables[idx].extend(diagnostics)

    def sample(self, cols, given=None, n=1):
        """ Draw samples from cols

        Parameters
        ----------
        cols : list(index)
            List of columns from which to jointly draw.
        given : list(tuple(int, value))
            List of column-value tuples that specify conditions
        n : int
            The number of samples to draw

        Example
        -------
        Draw whether an animal is fast and agile given that it is bulbous.
        >>> engine = Engine.load('examples/zoo.bcmodels')
        >>> engine.sample(['fast', 'agile'], given=[('bulbous': 1]))
        """
        # FIXME: make sure that given does not contain columns from cols
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

    def impute(self, col, rows=None):
        """ Infer the most likely value

        Note that confidence is not currently implemented and, for the time
        being, will always be `NaN`.

        Parameters
        ----------
        col : column name
            The column name to impute
        rows : list(row name), optional
            A list of the rows to impute. If None (default), all missing values
            will be imputed.

        Returns
        -------
        impdata : pandas.DataFrame
            Row-indexed DataFrame with a columns for the imputed values and
            the confidence (`conf`) in those values.
        """

        if rows is None:
            # Get row indices where col is null
            rows = self._df[pd.isnull(self._df[col])].index

        row_idxs = [self._converters['row2idx'][row] for row in rows]
        col_idx = self._converters['col2idx'][col]

        # FIXME: In the future we'll want a better way to determine
        # optimization bounds for different dtypes. If statements are gross.
        dtype = self._dtypes[col_idx]
        if dtype == 'continuous':
            lower = np.nanmin(self._data[:, col_idx])
            upper = np.nanmax(self._data[:, col_idx])
            bounds = (lower, upper,)
        elif dtype == 'categorical':
            bounds = self._converters['valmaps'][col]['val2idx'].values()
        else:
            raise ValueError('Unsupported dtype: {}'.format(dtype))

        impdata = []
        for row_idx in row_idxs:
            x, conf = mu.impute(row_idx, col_idx, self._models, bounds)
            if dtype == 'categorical':
                x = self._converters['valmaps'][col]['idx2val'][x]
            impdata.append({col: x, 'conf': conf})

        return pd.DataFrame(impdata, index=rows)

    def eval(self, testdata, metric=None):
        """ Impute and evaluate

        Parameters
        ----------
        testdata : pandas.Series
            Held-out values for a single column. Indices must be present in
            the baxcat data.
        metric : baxcat.metric.Metric
            The metric to use for evaluation
        """
        if not isinstance(testdata, pd.Series):
            raise TypeError('testdata must be a pandas.Series')

        if self._df[testdata.name].dtype == 'object':
            raise NotImplementedError('eval only works on numeric data')

        if metric is None:
            metric = metrics.SquaredError()

        impdata = self.impute(testdata.name, rows=testdata.index)

        return impdata, metric(testdata, impdata[testdata.name])

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

        Example
        -------
        The probability that an animal is fast and agile given that it is
        bulbous.
        >>> engine = Engine.load('examples/zoo.bcmodels')
        >>> engine.probability(np.array([[1, 1]]), ['fast', 'agile'],
        ...                    given=[('bulbous': 1]))
        """
        # TODO: make sure that given goes not caontain columns rom cols
        x = du.format_query_data(x)
        col_idxs = [self._converters['col2idx'][col] for col in cols]
        x_cnv = du.convert_data(x, cols, self._dtypes, self._converters)

        if given is not None:
            given = du.convert_given(given, self._dtypes, self._converters)

        return mu.probability(x_cnv, self._models, col_idxs)

    def predict(self, cols, given=None):
        """ Predict the value of columns given hypothetical values of other
        columns.
        """
        raise NotImplementedError

    def surprisal(self, col, rows=None):
        """ Surprisal, or self-information, of the observations in a column.

        Ignores missing values.

        Parameters
        ----------
        col : index
            The column index
        rows : list(index)
            A list of rows for which to compute surprisal. If not defined
            (default), computes for all rows.

        Returns
        -------
        pandas.DataFrame
            colums for 'column', 'row', 'value', and 'surprisal'
        """
        col_idx = self._converters['col2idx'][col]
        if rows is None:
            row_idxs = list(range(self._n_rows))
        else:
            row_idxs = [self._converters['row2idx'][row] for row in rows]

        vals = []
        rows = []
        queries = []
        for row_idx in row_idxs:
            row = self._converters['idx2row'][row_idx]
            x = self._data[row_idx, col_idx]

            # ignore missing values
            if not np.isnan(x):
                rows.append(row)
                vals.append(self._df[col][row])
                queries.append((row_idx, x,))

        s = mu.surprisal(col_idx, queries, self._models)
        data = []
        for val, si in zip(vals, s):
            data.append({'surprisal': si, col: val})

        return pd.DataFrame(data, index=rows)

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

    def row_similarity(self, row_a, row_b, wrt=None):
        """ The similarity between two rows in terms of their partitions. """
        if wrt is not None:
            colidxs = [self._converters['col2idx'][col] for col in wrt]

        if row_a == row_b:
            # XXX: we will assume that the user meant to do this
            return 1.0

        idx_a = self._converters['row2idx'][row_a]
        idx_b = self._converters['row2idx'][row_b]

        sim = np.zeros(self._n_models)
        for midx, model in enumerate(self._models):
            if wrt is not None:
                relviews = set([model['col_assignment'][c] for c in colidxs])
            else:
                relviews = list(range(len(model['row_assignments'])))

            for vidx in relviews:
                asgn = model['row_assignments'][vidx]
                sim[midx] += (asgn[idx_a] == asgn[idx_b])

            sim[midx] /= float(len(relviews))

        return np.mean(sim)

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
            if model['col_assignment'][idx_a] == \
                    model['col_assignment'][idx_b]:
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

    def linfoot(self, col_a, col_b, n_samples=1000):
        """ Wrapper for mutual_information(..., linfoot=True) """
        lfi = self.mutual_information(col_a, col_b, linfoot=True,
                                      n_samples=n_samples)
        return lfi

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
    def pairwise_func(self, func, idxs=None, **kwargs):
        """ Do a function over all paris of columns/rows

        Parameters
        ----------
        func : str
            `dependence_probability`, `mutual_information`, `linfoot`,
            or `conditional_entropy`
        idxs : list(column or row index), optional
            List of columns/rows. If None (default), the function is run on all
            pairs of columns/rows.
        n_samples : int, optional
            The number of samples for Monte Carlo approximation when
            applicable.
        """
        itertype = Engine.pairwise_iter_type(func)
        functype = Engine.pairwise_func_type(func)

        pfunc = getattr(self, func)

        if idxs is None:
            if functype == 'row':
                idxs = self._row_names
            elif functype == 'col':
                idxs = self._col_names
            else:
                msg = 'Unexpected functype ({}} for func {}'
                raise ValueError(msg.format(functype, func))

        mat = np.eye(len(idxs))
        if itertype == 'comb':
            for i, idx_a in enumerate(idxs):
                for j in range(i+1, len(idxs)):
                    idx_b = idxs[j]
                    fval = pfunc(idx_a, idx_b, **kwargs)
                    mat[i, j] = fval
                    mat[j, i] = fval
        elif itertype == 'prod':
            for i, idx_a in enumerate(idxs):
                for j, idx_b in enumerate(idxs):
                    fval = pfunc(idx_a, idx_b, **kwargs)
                    mat[i, j] = fval
        else:
            msg = 'Unexpcted itertype ({}) for func {}'
            raise ValueError(msg.format(itertype, func))

        df = pd.DataFrame(mat, index=idxs, columns=idxs)

        return df

    def heatmap(self, func, ignore_idxs=None, include_idxs=None,
                plot_kwargs=None, **kwargs):
        """ Heatmap of a pairwise function

        Prameters
        ---------
        func : str
            'dependence_probability', 'linfoot', 'mutual_information',
            'conditional_entropy', or 'row_similarity'
        n_samples : int
            the number of samples to estimate information theoretic quantities.
        ignore_idxs : list(column or row names)
            A list of column/row names not to include in the output.
        include_idxs : list(column or row names)
            A list of columns/rows to include in the output.
        """
        if plot_kwargs is None:
            plot_kwargs = {}

        idxs = None
        if ignore_idxs is not None:
            functype = Engine.pairwise_func_type(func)
            if functype == 'row':
                ptl_idxs = self._row_names
            elif functype == 'col':
                ptl_idxs = self._col_names
            else:
                msg = 'Unexpected functype ({}} for func {}'
                raise ValueError(msg.format(functype, func))

            idxs = [idx for idx in ptl_idxs if idx not in ignore_idxs]

        elif include_idxs is not None:
            idxs = include_idxs

        if func in ['dependence_probability', 'row_similarity']:
            plot_kwargs['vmin'] = plot_kwargs.get('vmin', 0.)
            plot_kwargs['vmax'] = plot_kwargs.get('vmax', 1.)
            plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'gray_r')
        elif func in ['mutual_information', 'linfoot']:
            plot_kwargs['cmap'] = plot_kwargs.get('cmap', 'gray_r')

        df = self.pairwise_func(func, idxs=idxs, **kwargs)

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

    def plot_state(self, state_idx, hl_rows=(), hl_cols=()):
        """ Visualize a cross-categorization state.

        .. note :: Work in progress
            This function is currently only suited to small data tables. There
            are problems with the row labels overlapping. There are problems
            with singleton views and categories having negative size or
            appearing as lines. Lots to fix.
        """
        if hl_rows != ():
            if not isinstance(hl_rows, (list, np.ndarray,)):
                hl_rows = [hl_rows]
            hl_rows = [self._converters['row2idx'][row] for row in hl_rows]

        if hl_cols != ():
            if not isinstance(hl_cols, (list, np.ndarray,)):
                hl_cols = [hl_cols]
            hl_cols = [self._converters['col2idx'][col] for col in hl_cols]

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
                         self._df.columns, hl_rows=hl_rows, hl_cols=hl_cols)

    # --- static utility functions
    @staticmethod
    def pairwise_func_type(func):
        functypes = {
            'dependence_probability': 'col',
            'mutual_information': 'col',
            'linfoot': 'col',
            'conditional_entropy': 'col',
            'row_similarity': 'row'}

        return functypes[func]

    @staticmethod
    def pairwise_iter_type(func):
        itertypes = {
            'dependence_probability': 'comb',
            'mutual_information': 'comb',
            'linfoot': 'comb',
            'conditional_entropy': 'prod',
            'row_similarity': 'comb'}

        return itertypes[func]

import sys
import baxcat.engine
import random
import numpy as np

from baxcat.state import BCState
from math import log


def printnow(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


class MInER(baxcat.engine.Engine):
    """
    WRITEME: Description

    Atributes
    ---------
    df : csv file name or pandas.DataFrame
        The input data
    logcf : function(row_index, data)
        Function that takes a row index and a list of data and calculates the
        log probability under some external model of the data coming from
        row_index.
    cols : list(col_index)
        List of the column indices, in order, that we want to do inference
        over. Each entry in the data array passed to `logcf` corresponds to a
        entry in `cols`.
    """
    def __init__(self, df, logcf, cols, metadata=None, **kwargs):
        super().__init__(df, metadata=metadata, **kwargs)

        self._miner_cols = cols
        self._miner_col_idxs = [self._converters['col2idx'][c] for c in cols]
        self._logcf = self._converter_logcf(logcf)

    def fit(self, n_iter=1, n_samples=10):
        rowlst = [i for i in range(self._n_rows)]
        for i in range(n_iter):
            printnow('+')
            self._init_bcstates()
            random.shuffle(rowlst)
            for row_idx in rowlst:
                self.resample_row(row_idx, n_samples)
                printnow('.')
            self._teardown_bcstates()
            self.run(1)
            printnow('%d of %d\n' % (i+1, n_iter,))

    def resample_row(self, row_idx, n_samples=1):
        assert(self._bcstates is not None)
        # build query indices
        qidxs = [[row_idx, col_idx] for col_idx in self._miner_col_idxs]

        x = [self._data[row_idx, c] for c in self._miner_col_idxs]
        assert(not np.any(np.isnan(x)))
        lp = self._logcf(row_idx, x)
        for _ in range(n_samples):
            # XXX: because the BCStates are acting as a prior, and they are our
            # transition function, they cancel out of the MH ratio. We only
            # need to worry about logcf
            x_pr = self._get_new_data(qidxs)
            assert(not np.any(np.isnan(x_pr)))
            lp_pr = self._logcf(row_idx, x_pr)
            if log(np.random.rand()) < lp_pr - lp:
                x = x_pr
                lp = lp_pr
                for bcs in self._bcstates:
                    bcs.replace_data(qidxs, x)
            # TODO: take MInER diagnostics?
        self._update_data(qidxs, x)

    def _get_new_data(self, qidxs):
        # uniform random choice corresponds to uniform weights
        bcs = random.choice(self._bcstates)
        return bcs.predictive_draw(qidxs)[0]

    def _init_bcstates(self):
        self._bcstates = []
        for model in self._models:
            sd = np.random.randint(2**31-1)
            init_kwarg = {'dtypes': self._dtypes,
                          'distargs': self._distargs,
                          'Zv': model['col_assignment'],
                          'Zrcv': model['row_assignments'],
                          'col_hypers': model['col_hypers'],
                          'state_alpha': model['state_alpha'],
                          'view_alphas': model['view_alphas'],
                          'seed': sd}
            self._bcstates.append(BCState(self._data.T, **init_kwarg))

    def _teardown_bcstates(self):
        self._model = [bcs.get_metadata() for bcs in self._bcstates]
        self._bcstates = None

    def _converter_logcf(self, func):
        """ Convert data sent to logcf to its values in df """
        def func_out(row_idx, data):
            row = self._converters['idx2row'][row_idx]
            data_cnv = []
            for col_idx, x in zip(self._miner_col_idxs, data):
                if self._dtypes[col_idx] == 'categorical':
                    col = self._converters['idx2col'][col_idx]
                    y = self._converters['valmaps'][col]['idx2val'][x]
                    data_cnv.append(y)
                else:
                    data_cnv.append(x)
            return func(row, data_cnv)

        return lambda row_idx, data: func_out(row_idx, data)

    def _update_data(self, idxs, data):
        for (row_idx, col_idx,), x in zip(idxs, data):
            assert not np.isnan(x)
            self._data[row_idx, col_idx] = x

            col = self._converters['idx2col'][col_idx]
            row = self._converters['idx2row'][row_idx]
            # TODO: Converter functions for each column instead of hacky crap!
            if self._dtypes[col_idx] == 'categorical':
                y = self._converters['valmaps'][col]['idx2val'][x]
            else:
                y = x

            self._df.loc[row, col] = y

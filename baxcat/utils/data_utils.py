""" Various utilities to process data for baxcat.

Data Format
-----------
Pass data as a pandas DataFrame.
Add metadata for each column with a dict indexed column name.

Example
-------
{
    'col_1': {
        'dtype': 'categorical',
        'values' : (1, 2, 3, 4, 99,)},
    'col_2': {
        'dtype': 'continuous'},
    'col_3': {
        'dtype': 'categorical',
        'values': ('disagree', 'agree', 'not sure',)}}
"""

import numpy as np
import pandas as pd
import math


def convert_given(given, dtypes, converters):
    g_out = []
    for col, val in given:
        col_idx = converters['col2idx'][col]
        if dtypes[col_idx] == 'categorical':
            val = converters['valmaps'][col]['val2idx'][val]
        g_out.append((col_idx, val,))

    return g_out


def convert_data(data, cols, dtypes, converters, to_val=False):
    assert data.shape[1] == len(cols)
    assert len(cols) <= len(dtypes)

    if to_val:
        def valmap(col, idx):
            return converters['valmaps'][col]['idx2val'][idx]
    else:
        def valmap(col, val):
            return converters['valmaps'][col]['val2idx'][val]

    # XXX: dtype of data out array should be object because it might have
    # strings in it if to_idx is True.
    data_out = np.zeros(data.shape, dtype=object)
    for i, col in enumerate(cols):
        col_idx = converters['col2idx'][col]
        if dtypes[col_idx] == 'categorical':
            x = data[:, i]
            xj = np.array([valmap(col, xi) for xi in x])
        else:
            xj = np.array(data[:, i], dtype=np.float64)
        data_out[:, i] = xj

    return data_out


def process_dataframe(df, n_models, metadata=None, n_unique_cutoff=20,
                      subset_size=None):
    """ Process data and generate metadata for use with c++ backend """

    col2idx = dict((c, ix) for ix, c in enumerate(df.columns))
    idx2col = dict((ix, c) for ix, c in enumerate(df.columns))

    dtypes = guess_dtypes(df, n_unique_cutoff=n_unique_cutoff,
                          metadata=metadata)

    if metadata is not None:
        for c, v in metadata.items():
            dtypes[col2idx[c]] = v['dtype']
    else:
        metadata = dict()

    valmaps = gen_valmaps(df, dtypes, metadata)
    data_array = dataframe_to_array(df, valmaps)
    distargs = []
    for col in df.columns:
        if col in valmaps:
            distargs.append([len(valmaps[col]['val2idx'])])
        else:
            distargs.append([0])

    converters = {
        'col2idx': col2idx,
        'idx2col': idx2col,
        'valmaps': valmaps}

    return data_array, dtypes, distargs, converters


def gen_row_idx_converters(df_index, n_models, subset_size=None):

    if subset_size is None:
        row2idx = [dict((r, ix) for ix, r in enumerate(df_index))]*n_models
        idx2row = [dict((ix, r) for ix, r in enumerate(df_index))]*n_models
    else:
        n_rows = len(df_index)
        row_subsets = gen_subset_indices(n_rows, subset_size, n_models)
        row2idx = [dict((r, df_index[r]) for r in rs) for rs in row_subsets]
        idx2row = [dict((df_index[r], r) for r in rs) for rs in row_subsets]

    return row2idx, idx2row


def dataframe_to_array(df, valmaps):
    data = np.zeros(df.shape, dtype=float)
    for cidx, col in enumerate(df.columns):
        if col in valmaps:
            coldata = np.zeros(df.shape[0])
            val2idx = valmaps[col]['val2idx']
            for ridx, val in enumerate(df[col].values):
                if pd.isnull(val):
                    coldata[ridx] = float('NaN')
                else:
                    coldata[ridx] = val2idx[val]
        else:
            coldata = df[col].values

        data[:, cidx] = coldata

    data = np.array(data, dtype=float)
    return data


def gen_valmaps(df, dtypes, metadata):
    """ FIXME: Write """
    valmaps = dict()
    for idx, dtype in enumerate(dtypes):
        if dtype == 'categorical':
            col = df.columns[idx]
            if col in metadata:
                vals = np.sort(metadata[col]['values'])
            else:
                vals = np.sort(df[col].dropna().unique())

            val2idx = dict((val, idx) for idx, val in enumerate(vals))
            idx2val = dict((idx, val) for idx, val in enumerate(vals))

            valmaps[col] = dict()
            valmaps[col]['idx2val'] = idx2val
            valmaps[col]['val2idx'] = val2idx

    return valmaps


def guess_dtypes(df, n_unique_cutoff=20, metadata=None):
    """
    Guess the datatypes for the columns in the pandas DataFrame, df. Can only
    guess continuous and categorical data types.

    Parameters
    ----------
    df : pandas.DataFrame
        The data
    n_unique_cutoff : int
        The maximum number of unique values in numeric column before it is
        considered continuous.
    metadata : dict
        baxcat feature metadata.

    Returns
    -------
    dtypes : list(str)
        List of the data type for each column.
    """
    if metadata is None:
        metadata = {}

    dtypes = []
    for col in df:
        if col in metadata:
            dtype = metadata[col]['dtype']
        else:
            n_unique = len(df[col].unique())
            dtype = str(df[col].dtype)
            if np.issubdtype(dtype, np.number) and n_unique > n_unique_cutoff:
                dtype = 'continuous'
            else:
                dtype = 'categorical'

        dtypes.append(dtype)

    return dtypes


def format_query_data(data):
    """ converts query data to 2D numpy array ready for use with queries. """

    if not isinstance(data, (list, np.ndarray,)):
        data_out = np.array([[data]], dtype=object)
    elif isinstance(data, list):
        if isinstance(data[0], (list, np.ndarray,)):
            data_out = np.array(data, dtype=object)
        else:
            data_out = np.array([data], dtype=object)
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            data_out = np.array([data], dtype=object)
        else:
            data_out = data
    else:
        raise TypeError("I do not know how to handle a %s." % type(data))

    assert len(data_out.shape) == 2

    return data_out


def gen_subset_indices(n_rows, subset_size, n_sets):
    """ Generate subset indices and converters """
    if subset_size > 1.0:
        raise ValueError('subset_size must be no greater than 1.0')

    if subset_size < 1.0/n_sets:
        raise ValueError("subset_size too small, must cover entire dataset")

    n_subset = math.ceil(n_rows*subset_size)
    row_idxs = list(range(n_rows))
    np.random.shuffle(row_idxs)

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    subsets = list(chunks(row_idxs, int(n_rows/n_sets)))

    for subset in subsets:
        if len(subset) < n_subset:
            other_rows = [i for i in range(n_rows) if i not in subset]
            np.random.shuffle(other_rows)
            k = n_subset - len(subset)
            subset.extend(other_rows[:k])

        assert len(subset) == n_subset

        subset.sort()

    return subsets

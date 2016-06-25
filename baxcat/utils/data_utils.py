
import numpy as np
import pandas as pd

"""
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
        valmap = lambda col, idx: converters['valmaps'][col]['idx2val'][idx]
    else:
        valmap = lambda col, val: converters['valmaps'][col]['val2idx'][val]

    # XXX: dtype of data out array should be object because it might have
    # strings in it if to_idx is True.
    data_out = np.zeros(data.shape, dtype=object)
    for i, col in enumerate(cols):
        col_idx = converters['col2idx'][col]
        if dtypes[col_idx] == 'categorical':
            x = data[:, i]
            xj = np.array([valmap(col, xi) for xi in x])
        else:
            xj = data[:, i]
        data_out[:, i] = xj

    return data_out


def process_dataframe(df, metadata=None, n_unique_cutoff=20):
    """ Process data and generate metadata for use with c++ backend """

    col2idx = dict((col, idx) for idx, col in enumerate(df.columns))
    idx2col = dict((idx, col) for idx, col in enumerate(df.columns))

    row2idx = dict((row, idx) for idx, row in enumerate(df.index))
    idx2row = dict((idx, row) for idx, row in enumerate(df.index))

    dtypes = guess_dtypes(df, n_unique_cutoff=n_unique_cutoff)

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
        'row2idx': row2idx,
        'idx2row': idx2row,
        'valmaps': valmaps}

    return data_array, dtypes, distargs, converters


def dataframe_to_array(df, valmaps):
    data = np.zeros(df.shape)
    n_rows = data.shape[0]
    for cidx, col in enumerate(df.columns):
        if col in valmaps:
            for ridx in range(n_rows):
                if pd.isnull(df.ix[ridx, cidx]):
                    data[ridx, cidx] = df.ix[ridx, cidx]
                else:
                    data[ridx, cidx] = valmaps[col]['val2idx'][df.ix[ridx, cidx]]
        else:
            for ridx in range(n_rows):
                data[ridx, cidx] = df.ix[ridx, cidx]

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


def guess_dtypes(df, n_unique_cutoff=20):
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

    Returns
    -------
    dtypes : list(str)
        List of the data type for each column.
    """
    dtypes = []
    for col in df:
        if 'float' in str(df[col].dtype)and len(df[col].unique()) > n_unique_cutoff:
            dtype = 'continuous'
        else:
            dtype = 'categorical'
        dtypes.append(dtype)

    return dtypes

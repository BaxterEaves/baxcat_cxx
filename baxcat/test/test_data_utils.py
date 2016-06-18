import pytest
import pandas as pd
import numpy as np

from baxcat.utils import data as du


@pytest.fixture()
def smalldf():

    s1 = pd.Series(np.random.rand(30))
    s2 = pd.Series([0.0, 1.0]*15)
    s3 = pd.Series(['one', 'two', 'three']*10)
    s4 = pd.Series(np.random.rand(30))

    return pd.concat([s1, s2, s3, s4], axis=1)


# Guess data types
# `````````````````````````````````````````````````````````````````````````````
def test_guess_dtypes_should_guess_correct_types_continuous():
    df = pd.DataFrame(np.random.rand(30, 4))
    dtypes = du.guess_dtypes(df)

    assert all([dt == 'continuous' for dt in dtypes])


def test_guess_dtypes_should_guess_correct_types_continuous_short():
    # small number of unique values
    df = pd.DataFrame(np.random.rand(5, 4))
    dtypes = du.guess_dtypes(df)

    assert all([dt == 'categorical' for dt in dtypes])


def test_guess_dtypes_increase_unique_vals_cutoff():
    # large number of unique values
    df = pd.DataFrame(np.random.rand(30, 4))
    dtypes = du.guess_dtypes(df, n_unique_cutoff=32)

    assert all([dt == 'categorical' for dt in dtypes])


def test_guess_dtypes_decrease_unique_vals_cutoff():
    # large number of unique values
    df = pd.DataFrame(np.random.rand(5, 4))
    dtypes = du.guess_dtypes(df, n_unique_cutoff=2)

    assert all([dt == 'continuous' for dt in dtypes])


def test_guess_dtypes_mixed_types(smalldf):
    dtypes = du.guess_dtypes(smalldf)

    assert dtypes[0] == 'continuous'
    assert dtypes[1] == 'categorical'
    assert dtypes[2] == 'categorical'
    assert dtypes[3] == 'continuous'


def test_guess_dtypes_mixed_types_missing_vals(smalldf):

    smalldf.ix[0, 0] = float('NaN')
    smalldf.ix[0, 1] = float('NaN')
    smalldf.ix[0, 2] = float('NaN')

    dtypes = du.guess_dtypes(smalldf)

    assert dtypes[0] == 'continuous'
    assert dtypes[1] == 'categorical'
    assert dtypes[2] == 'categorical'
    assert dtypes[3] == 'continuous'


# valmaps
# `````````````````````````````````````````````````````````````````````````````
def test_gen_valmaps_default(smalldf):
    dtypes = ['continuous', 'categorical', 'categorical', 'continuous']
    metadata = dict()
    valmaps = du.gen_valmaps(smalldf, dtypes, metadata)

    assert 0 not in valmaps
    assert 3 not in valmaps

    assert 1 in valmaps
    assert 2 in valmaps

    valmap_1 = valmaps[1]
    valmap_2 = valmaps[2]

    assert len(valmap_1['val2idx']) == 2
    assert len(valmap_1['idx2val']) == 2
    assert valmap_1['val2idx'][0] == 0
    assert valmap_1['val2idx'][1] == 1
    assert valmap_1['idx2val'][0] == 0
    assert valmap_1['idx2val'][1] == 1

    assert len(valmap_2['val2idx']) == 3
    assert len(valmap_2['idx2val']) == 3
    assert valmap_2['val2idx']['one'] == 0
    assert valmap_2['val2idx']['three'] == 1
    assert valmap_2['val2idx']['two'] == 2
    assert valmap_2['idx2val'][0] == 'one'
    assert valmap_2['idx2val'][1] == 'three'
    assert valmap_2['idx2val'][2] == 'two'


def test_gen_valmaps_metadata(smalldf):
    dtypes = ['continuous', 'categorical', 'categorical', 'continuous']
    metadata = {}
    metadata[1] = {
        'dtype': 'categorical',
        'values': [-1, 0, 1, 2, 99]}
    metadata[2] = {
        'dtype': 'categorical',
        'values': ['zero', 'one', 'two', 'three']}

    valmaps = du.gen_valmaps(smalldf, dtypes, metadata)

    assert 0 not in valmaps
    assert 3 not in valmaps

    assert 1 in valmaps
    assert 2 in valmaps

    valmap_1 = valmaps[1]
    valmap_2 = valmaps[2]

    assert len(valmap_1['val2idx']) == 5
    assert len(valmap_1['idx2val']) == 5

    # col 2
    assert valmap_1['val2idx'][-1] == 0
    assert valmap_1['val2idx'][0] == 1
    assert valmap_1['val2idx'][1] == 2
    assert valmap_1['val2idx'][2] == 3
    assert valmap_1['val2idx'][99] == 4

    assert valmap_1['idx2val'][0] == -1
    assert valmap_1['idx2val'][1] == 0
    assert valmap_1['idx2val'][2] == 1
    assert valmap_1['idx2val'][3] == 2
    assert valmap_1['idx2val'][4] == 99

    # col 2
    assert valmap_2['val2idx']['one'] == 0
    assert valmap_2['val2idx']['three'] == 1
    assert valmap_2['val2idx']['two'] == 2
    assert valmap_2['val2idx']['zero'] == 3

    assert valmap_2['idx2val'][0] == 'one'
    assert valmap_2['idx2val'][1] == 'three'
    assert valmap_2['idx2val'][2] == 'two'
    assert valmap_2['idx2val'][3] == 'zero'


def test_gen_valmaps_default_missing_vals(smalldf):
    # Missing value NaN should not appear in the value map
    dtypes = ['continuous', 'categorical', 'categorical', 'continuous']
    metadata = dict()
    smalldf.ix[0, 1] = float('NaN')
    smalldf.ix[1, 2] = float('NaN')
    valmaps = du.gen_valmaps(smalldf, dtypes, metadata)

    assert 0 not in valmaps
    assert 3 not in valmaps

    assert 1 in valmaps
    assert 2 in valmaps

    valmap_1 = valmaps[1]
    valmap_2 = valmaps[2]

    assert len(valmap_1['val2idx']) == 2
    assert len(valmap_1['idx2val']) == 2
    assert valmap_1['val2idx'][0] == 0
    assert valmap_1['val2idx'][1] == 1
    assert valmap_1['idx2val'][0] == 0
    assert valmap_1['idx2val'][1] == 1

    assert len(valmap_2['val2idx']) == 3
    assert len(valmap_2['idx2val']) == 3
    assert valmap_2['val2idx']['one'] == 0
    assert valmap_2['val2idx']['three'] == 1
    assert valmap_2['val2idx']['two'] == 2
    assert valmap_2['idx2val'][0] == 'one'
    assert valmap_2['idx2val'][1] == 'three'
    assert valmap_2['idx2val'][2] == 'two'


# dataframe_to_array
# `````````````````````````````````````````````````````````````````````````````
def test_dataframe_to_array_all_continuous():
    n_cols = 5
    df = pd.DataFrame(np.random.rand(30, n_cols))
    valmaps = dict()

    data = du.dataframe_to_array(df, valmaps)

    assert data.shape == df.shape
    assert 'float' in str(data.dtype)


def test_dataframe_to_array_all_categorical():
    s_1 = pd.Series([-1, 0, 2, 1])
    s_2 = pd.Series(['one', 'two', 'three', 'four'])
    df = pd.concat([s_1, s_2], axis=1)

    dtypes = ['categorical']*2
    metadata = dict()

    valmaps = du.gen_valmaps(df, dtypes, metadata)
    data = du.dataframe_to_array(df, valmaps)

    assert data.shape == df.shape
    assert 'float' in str(data.dtype)

    assert data[0, 0] == 0
    assert data[1, 0] == 1
    assert data[2, 0] == 3
    assert data[3, 0] == 2

    assert data[0, 1] == 1
    assert data[1, 1] == 3
    assert data[2, 1] == 2
    assert data[3, 1] == 0


def test_dataframe_to_array_all_categorical_with_missing_vals():
    s_1 = pd.Series([-1, 0, 2, 1, float('NaN')])
    s_2 = pd.Series(['one', 'two', 'three', 'four', float('NaN')])
    df = pd.concat([s_1, s_2], axis=1)

    dtypes = ['categorical']*2
    metadata = dict()

    valmaps = du.gen_valmaps(df, dtypes, metadata)
    data = du.dataframe_to_array(df, valmaps)

    assert data.shape == df.shape
    assert 'float' in str(data.dtype)

    assert data[0, 0] == 0
    assert data[1, 0] == 1
    assert data[2, 0] == 3
    assert data[3, 0] == 2
    assert np.isnan(data[4, 0])

    assert data[0, 1] == 1
    assert data[1, 1] == 3
    assert data[2, 1] == 2
    assert data[3, 1] == 0
    assert np.isnan(data[4, 1])


# test process_dataframe
# `````````````````````````````````````````````````````````````````````````````
def test_process_dataframe_output_default(smalldf):
    data_array, dtypes, distargs, converters = du.process_dataframe(smalldf)

    assert data_array.shape == smalldf.shape
    assert 'float' in str(data_array.dtype)

    assert dtypes[0] == 'continuous'
    assert dtypes[1] == 'categorical'
    assert dtypes[2] == 'categorical'
    assert dtypes[3] == 'continuous'

    # distargs for continuous is irrelevant, distargs for categorical should be
    # the number of values
    assert len(distargs[0]) == 1  # continuous
    assert distargs[0][0] == 0
    assert len(distargs[1]) == 1  # categorical
    assert distargs[1][0] == 2
    assert len(distargs[2]) == 1  # categorical
    assert distargs[2][0] == 3
    assert len(distargs[3]) == 1  # continuous
    assert distargs[3][0] == 0

    # converters
    assert len(converters['col2idx']) == smalldf.shape[1]
    assert len(converters['idx2col']) == smalldf.shape[1]
    assert len(converters['row2idx']) == smalldf.shape[0]
    assert len(converters['idx2row']) == smalldf.shape[0]
    assert len(converters['valmaps']) == 2  # number of categorical cols

import pytest
import pandas as pd
import numpy as np
import itertools as it

from baxcat.utils import data_utils as du


@pytest.fixture()
def smalldf():

    s1 = pd.Series(np.array(np.random.rand(30)*2**31, dtype=int))
    s2 = pd.Series([0.0, 1.0]*15)
    s3 = pd.Series(['one', 'two', 'three']*10)
    s4 = pd.Series(np.random.rand(30))

    return pd.concat([s1, s2, s3, s4], axis=1)


# Guess data types
# ---
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
# ---
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
# ---
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
# ---
def test_process_dataframe_output_default(smalldf):
    data_array, dtypes, distargs, converters = du.process_dataframe(smalldf, 2)

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
    assert len(converters['valmaps']) == 2  # number of categorical cols


# converters
# ---
def test_convert_continuous_given_should_change_cols_not_vals():
    given_in = [('x', 1.2,), ('y', 3.4,), ('y', 2,)]
    dtypes = ['continuous']*2
    converters = {
        'col2idx': {'x': 0, 'y': 1},
        'valmaps': {}}

    given_out = du.convert_given(given_in, dtypes, converters)

    for (k1, v1), (k2, v2) in zip(given_in, given_out):
        assert k1 != k2
        assert v1 == v2
        assert isinstance(k1, str)
        assert isinstance(k2, int)

    assert given_out[0] == (0, 1.2,)
    assert given_out[1] == (1, 3.4,)
    assert given_out[2] == (1, 2,)


def test_convert_categorical_given_should_change_cols_and_vals():
    given_in = [('x', 'a',), ('y', 'bb',), ('y', 'cc',)]
    dtypes = ['categorical']*2
    converters = {
        'col2idx':
            {'x': 0, 'y': 1},
        'valmaps': {
            'x': {
                'val2idx': {'a': 0, 'b': 1, 'c': 2}},
            'y': {
                'val2idx': {'aa': 0, 'bb': 1, 'cc': 2}}}}

    given_out = du.convert_given(given_in, dtypes, converters)

    assert given_out[0] == (0, 0,)
    assert given_out[1] == (1, 1,)
    assert given_out[2] == (1, 2,)


def test_convert_mixed_given_output():
    given_in = [('x', 1.2,), ('y', 'bb',), ('y', 'cc',)]
    dtypes = ['continuous', 'categorical']
    converters = {
        'col2idx':
            {'x': 0, 'y': 1},
        'valmaps': {
            'y': {
                'val2idx': {'aa': 0, 'bb': 1, 'cc': 2}}}}

    given_out = du.convert_given(given_in, dtypes, converters)

    assert given_out[0] == (0, 1.2,)
    assert given_out[1] == (1, 1,)
    assert given_out[2] == (1, 2,)


def test_convert_continuous_data_should_do_nothing():
    data_in = np.random.rand(10, 2)
    dtypes = ['continuous']*2
    converters = {
        'col2idx': {'x': 0, 'y': 1},
        'valmaps': {}}

    data_out = du.convert_data(data_in, ['x', 'y'], dtypes, converters)

    assert data_in.shape == data_out.shape

    n_rows, n_cols = data_in.shape
    for i, j in it.product(range(n_rows), range(n_cols)):
        assert data_in[i, j] == data_out[i, j]

    data_out = du.convert_data(data_in, ['x', 'y'], dtypes, converters,
                               to_val=True)

    assert data_in.shape == data_out.shape

    for i, j in it.product(range(n_rows), range(n_cols)):
        assert data_in[i, j] == data_out[i, j]


def test_convert_categorical_data_should_change_everything():
    data_in = np.array([
        ['a', 3],
        ['b', 2],
        ['c', 2],
        ['a', 1],
        ['b', 1]], dtype=object)
    dtypes = ['categorical']*2
    converters = {
        'col2idx':
            {'x': 0, 'y': 1},
        'valmaps': {
            'x': {
                'val2idx': {'a': 0, 'b': 1, 'c': 2},
                'idx2val': {0: 'a', 1: 'b', 2: 'c'}},
            'y': {
                'val2idx': {1: 0, 2: 1, 3: 2},
                'idx2val': {0: 1, 1: 2, 2: 3}}}}

    data_out = du.convert_data(data_in, ['x', 'y'], dtypes, converters)

    assert data_out.shape == data_in.shape

    assert data_out[0, 0] == 0
    assert data_out[1, 0] == 1
    assert data_out[2, 0] == 2
    assert data_out[3, 0] == 0
    assert data_out[4, 0] == 1

    assert data_out[0, 1] == 2
    assert data_out[1, 1] == 1
    assert data_out[2, 1] == 1
    assert data_out[3, 1] == 0
    assert data_out[4, 1] == 0

    data_in_2 = du.convert_data(data_out, ['x', 'y'], dtypes, converters,
                                to_val=True)

    assert data_in_2.shape == data_in.shape

    n_rows, n_cols = data_in.shape
    for i, j in it.product(range(n_rows), range(n_cols)):
        assert data_in[i, j] == data_in_2[i, j]


def test_convert_categorical_data_single_column():
    data_in = np.array([
        ['a', 3],
        ['b', 2],
        ['c', 2],
        ['a', 1],
        ['b', 1]], dtype=object)
    dtypes = ['categorical', 'continuous']
    converters = {
        'col2idx':
            {'x': 0, 'y': 1},
        'valmaps': {
            'x': {
                'val2idx': {'a': 0, 'b': 1, 'c': 2},
                'idx2val': {0: 'a', 1: 'b', 2: 'c'}}}}

    data_out = du.convert_data(data_in, ['x', 'y'], dtypes, converters)

    assert data_out.shape == data_in.shape

    assert data_out[0, 0] == 0
    assert data_out[1, 0] == 1
    assert data_out[2, 0] == 2
    assert data_out[3, 0] == 0
    assert data_out[4, 0] == 1

    assert data_out[0, 1] == 3
    assert data_out[1, 1] == 2
    assert data_out[2, 1] == 2
    assert data_out[3, 1] == 1
    assert data_out[4, 1] == 1

    data_in_2 = du.convert_data(data_out, ['x', 'y'], dtypes, converters,
                                to_val=True)

    assert data_in_2.shape == data_in.shape

    n_rows, n_cols = data_in.shape
    for i, j in it.product(range(n_rows), range(n_cols)):
        assert data_in[i, j] == data_in_2[i, j]


# ---
def test_format_query_data_should_return_2d_numpy_array_float():
    # single float data
    x = du.format_query_data(1.2)
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 1,)


def test_format_query_data_should_return_2d_numpy_array_str():
    # single string data
    x = du.format_query_data('string')
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 1,)


def test_format_query_data_should_return_2d_numpy_array_list():
    # single list
    x = du.format_query_data([1.2, 'string'])
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 2,)


def test_format_query_data_should_return_2d_numpy_array_float_array():
    # single float64 array
    x = du.format_query_data(np.array([1.2, 2.1]))
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 2,)


def test_format_query_data_should_return_2d_numpy_array_obj_array():
    # single object array
    x = du.format_query_data(np.array([.2, 'string']))
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 2,)


def test_format_query_data_should_return_2d_numpy_array_list_of_lists():
    # list of lists
    x = du.format_query_data([[.2, 'string'], [.1, 'x']])
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, 2,)


def test_format_query_data_should_return_2d_numpy_array_list_of_array():
    # list of arrays
    x = du.format_query_data([np.array([.2, 'string']), np.array([.1, 'x'])])
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, 2,)


def test_format_query_data_should_return_2d_numpy_array_2d_array():
    # 2d array
    x = du.format_query_data(np.eye(2))
    assert isinstance(x, np.ndarray)
    assert x.shape == (2, 2,)


# --
def test_gen_subset_indices():
    # generate 5 data sets each with half the data
    subsets = du.gen_subset_indices(20, .5, 5)

    assert len(subsets) == 5
    assert all(len(subset) == 10 for subset in subsets)

    # each element should be unique
    assert all(len(set(subset)) == 10 for subset in subsets)

    allidxs = []
    for subset in subsets:
        allidxs.extend(subset)

    assert len(set(allidxs)) == 20


def test_gen_subset_indices_raises_value_error_with_low_set_size():
    with pytest.raises(ValueError):
        du.gen_subset_indices(20, .1, 5)


def test_gen_subset_indices_raises_value_error_with_high_set_size():
    with pytest.raises(ValueError):
        du.gen_subset_indices(20, 1.2, 5)

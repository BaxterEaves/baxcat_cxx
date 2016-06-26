import pytest

import tempfile
import pandas as pd
import numpy as np

from baxcat.engine import Engine


@pytest.fixture()
def smalldf():

    s1 = pd.Series(np.random.rand(30))
    s2 = pd.Series([0.0, 1.0]*15)
    s3 = pd.Series(['one', 'two', 'three']*10)
    s4 = pd.Series(np.random.rand(30))

    df = pd.concat([s1, s2, s3, s4], axis=1)
    df.columns = ['x_1', 'x_2', 'x_3', 'x_4']
    return df


@pytest.fixture()
def smallengine(smalldf):
    engine = Engine(smalldf, no_mp=True)
    engine.init_models(2)
    engine.run(10)
    # print(engine.col_info())
    return engine


# test init
# `````````````````````````````````````````````````````````````````````````````
def test_engine_init_smoke_default(smalldf):
    df = pd.DataFrame(np.random.rand(30, 5))
    engine = Engine(df, no_mp=True)
    engine.init_models(1)

    engine = Engine(smalldf)
    engine.init_models(1)


def test_engine_init_smoke_metadata(smalldf):
    metadata = dict()
    metadata['x_2'] = {
        'dtype': 'categorical',
        'values': [-1, 0, 1, 99]}
    metadata['x_3'] = {
        'dtype': 'categorical',
        'values': ['zero', 'one', 'two', 'three', 'four']}

    engine = Engine(smalldf, metadata=metadata)
    engine.init_models(1)
    engine = Engine(smalldf,  metadata=metadata)
    engine.init_models(1)


# test run
# `````````````````````````````````````````````````````````````````````````````
def test_engine_run_smoke_default(smalldf):
    engine = Engine(smalldf)
    engine.init_models(1)
    engine.run()
    engine.run(10)
    assert len(engine.models) == 1


def test_engine_run_smoke_multiple(smalldf):
    engine = Engine(smalldf)
    engine.init_models(10)
    engine.run()
    engine.run(10)
    assert len(engine.models) == 10


def test_run_with_checkpoint_valid_diagnostic_output(smalldf):
    engine = Engine(smalldf)
    engine.init_models(5)
    engine.run(10, checkpoint=5)

    tables = engine._diagnostic_tables

    assert len(tables) == 5

    for table in tables:
        assert len(table) == 3
        for entry in table:
            assert 'log_score' in entry
            assert 'iters' in entry
            assert 'time' in entry


def test_run_on_model_subset_should_only_run_those_models(smalldf):
    engine = Engine(smalldf)
    engine.init_models(5)
    engine.run(10, checkpoint=5)
    engine.run(10, checkpoint=5, model_idxs=[1, 2])

    tables = engine._diagnostic_tables

    assert len(tables) == 5

    assert len(tables[0]) == 3
    assert len(tables[1]) == 5
    assert len(tables[2]) == 5
    assert len(tables[3]) == 3
    assert len(tables[4]) == 3


# save and load
# `````````````````````````````````````````````````````````````````````````````
def test_save_smoke(smalldf):
    engine = Engine(smalldf)
    engine.init_models(5)

    with tempfile.NamedTemporaryFile('wb') as tf:
        engine.save(tf.name)


def test_load_smoke(smalldf):
    engine = Engine(smalldf)
    engine.init_models(5)

    with tempfile.NamedTemporaryFile('wb') as tf:
        engine.save(tf.name)
        Engine.load(tf.name)


def test_save_and_load_equivalence(smalldf):
    engine = Engine(smalldf)
    engine.init_models(5)

    with tempfile.NamedTemporaryFile('wb') as tf:
        engine.save(tf.name)
        new_engine = Engine.load(tf.name)

        assert engine._models == new_engine._models
        assert engine._dtypes == new_engine._dtypes
        assert engine._metadata == new_engine._metadata
        assert engine._converters == new_engine._converters
        assert engine._diagnostic_tables == new_engine._diagnostic_tables
        assert all(engine._row_names == new_engine._row_names)
        assert all(engine._col_names == new_engine._col_names)


# dependence probability
# `````````````````````````````````````````````````````````````````````````````
def test_dependence_probability():
    x = np.random.randn(30)

    s1 = pd.Series(x)
    s2 = pd.Series(x + 1.0)
    s3 = pd.Series(np.random.rand(30))

    df = pd.concat([s1, s2, s3], axis=1)
    df.columns = ['c0', 'c1', 'c2']

    engine = Engine(df, no_mp=True)
    engine.init_models(20)
    engine.run(10)
    depprob_01 = engine.dependence_probability('c0', 'c1')
    depprob_02 = engine.dependence_probability('c0', 'c2')
    depprob_12 = engine.dependence_probability('c1', 'c2')

    assert depprob_01 > depprob_02
    assert depprob_01 > depprob_12


def test_pairwise_dependence_probability():
    x = np.random.randn(30)

    s1 = pd.Series(x)
    s2 = pd.Series(x + 1.0)
    s3 = pd.Series(np.random.rand(30))

    df = pd.concat([s1, s2, s3], axis=1)
    df.columns = ['c0', 'c1', 'c2']

    engine = Engine(df, no_mp=True)
    engine.init_models(10)
    engine.run(5)

    depprob = engine.pairwise_func('dependence_probability')
    assert depprob.ix[0, 0] == 1.
    assert depprob.ix[1, 1] == 1.
    assert depprob.ix[2, 2] == 1.

    assert depprob.ix[0, 1] == depprob.ix[1, 0]
    assert depprob.ix[0, 2] == depprob.ix[2, 0]
    assert depprob.ix[1, 2] == depprob.ix[2, 1]


# probability
# `````````````````````````````````````````````````````````````````````````````
def test_probability_single_col_single_datum(smallengine):
    data = 1.2
    col = 'x_1'
    p = smallengine.probability(data, [col])

    assert isinstance(p, (float, np.float64,))


def test_probability_multi_col_single_datum(smallengine):
    data = [1.2, 2.3]
    cols = ['x_1', 'x_4']
    p = smallengine.probability(data, cols)

    assert isinstance(p, (float, np.float64,))


def test_probability_single_col_multi_datum(smallengine):
    data = [[1.2], [0.2]]
    col = 'x_1'
    p = smallengine.probability(data, [col])

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


def test_probability_multi_col_multi_datum(smallengine):
    data = [[1.2, 'two'], [0.2, 'one']]
    cols = ['x_1', 'x_3']
    given = [('x_2', 0,), ('x_4', 0,)]
    p = smallengine.probability(data, cols, given=given)

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


def test_probability_single_col_single_datum_given(smallengine):
    data = 1.2
    col = 'x_1'
    given = [('x_2', 0,), ('x_4', 0,)]
    p = smallengine.probability(data, [col], given=given)

    assert isinstance(p, (float, np.float64,))


def test_probability_multi_col_single_datum_given(smallengine):
    data = [1.2, 2.3]
    cols = ['x_1', 'x_4']
    given = [('x_2', 0,), ('x_4', 0,)]
    p = smallengine.probability(data, cols, given=given)

    assert isinstance(p, (float, np.float64,))


def test_probability_single_col_multi_datum_given(smallengine):
    data = [[1.2], [0.2]]
    col = 'x_1'
    p = smallengine.probability(data, [col])

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


def test_probability_multi_col_multi_datum_given(smallengine):
    data = [[1.2, 'two'], [0.2, 'one']]
    cols = ['x_1', 'x_3']
    given = [('x_2', 0,), ('x_4', 0,)]
    p = smallengine.probability(data, cols, given=given)

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


# sample
# ````````````````````````````````````````````````````````````````````````````
def test_sample_single_col_single_float(smallengine):
    cols = ['x_1']
    x = smallengine.sample(cols, n=1)

    assert isinstance(x, (float, np.float64,))


def test_sample_single_col_single_str(smallengine):
    cols = ['x_3']
    x = smallengine.sample(cols, n=1)

    assert isinstance(x, str)


def test_sample_multi_col_single_mixed(smallengine):
    cols = ['x_1', 'x_3']
    x = smallengine.sample(cols, n=1)

    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)
    assert isinstance(x[0], (float, np.float64,))
    assert isinstance(x[1], str)


def test_sample_multi_col_multi_mixed(smallengine):
    cols = ['x_1', 'x_3']
    x = smallengine.sample(cols, n=3)

    assert isinstance(x, np.ndarray)
    assert x.shape == (3, 2,)
    assert isinstance(x[0, 0], (float, np.float64,))
    assert isinstance(x[1, 0], (float, np.float64,))
    assert isinstance(x[2, 0], (float, np.float64,))

    assert isinstance(x[0, 1], str)
    assert isinstance(x[1, 1], str)
    assert isinstance(x[2, 1], str)

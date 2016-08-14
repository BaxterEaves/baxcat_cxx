import pytest

import tempfile
import pandas as pd
import numpy as np

from baxcat.engine import Engine


def smalldf():
    s1 = pd.Series(np.random.rand(30))
    s2 = pd.Series([0.0, 1.0]*15)
    s3 = pd.Series(['one', 'two', 'three']*10)
    s4 = pd.Series(np.random.rand(30))

    df = pd.concat([s1, s2, s3, s4], axis=1)
    df.columns = ['x_1', 'x_2', 'x_3', 'x_4']
    return df


def smalldf_mssg():
    df = smalldf()
    df['x_1'].ix[0] = float('NaN')
    df['x_2'].ix[1] = float('NaN')
    df['x_3'].ix[2] = float('NaN')
    df['x_4'].ix[3] = float('NaN')

    return df


def gen_engine(df):
    engine = Engine(df, no_mp=True)
    engine.init_models(2)
    engine.run(10)
    # print(engine.col_info())
    return engine


# test init
# `````````````````````````````````````````````````````````````````````````````
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_engine_init_smoke_default(gendf):
    df = gendf()
    engine = Engine(df)
    engine.init_models(1)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_engine_init_smoke_metadata(gendf):
    df = gendf()

    metadata = dict()
    metadata['x_2'] = {
        'dtype': 'categorical',
        'values': [-1, 0, 1, 99]}
    metadata['x_3'] = {
        'dtype': 'categorical',
        'values': ['zero', 'one', 'two', 'three', 'four']}

    engine = Engine(df, metadata=metadata)
    engine.init_models(1)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_engine_init_structureless(gendf):
    df = gendf()

    df = pd.DataFrame(np.random.rand(30, 5))
    engine = Engine(df, no_mp=True)
    engine.init_models(4, structureless=True)

    assert len(engine._models) == 4
    assert all([max(m['col_assignment']) == 0 for m in engine._models])
    assert all([len(m['row_assignments']) == 1 for m in engine._models])
    for m in engine._models:
        assert all([max(z) == 0 for z in m['row_assignments']])


# test run
# `````````````````````````````````````````````````````````````````````````````
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_engine_run_smoke_default(gendf):
    df = gendf()

    engine = Engine(df)
    engine.init_models(1)
    engine.run()
    engine.run(10)

    assert len(engine.models) == 1


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_engine_run_smoke_multiple(gendf):
    df = gendf()

    engine = Engine(df)
    engine.init_models(10)
    engine.run()
    engine.run(10)

    assert len(engine.models) == 10


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_run_with_checkpoint_valid_diagnostic_output(gendf):
    df = gendf()

    engine = Engine(df)
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


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_run_on_model_subset_should_only_run_those_models(gendf):
    df = gendf()

    engine = Engine(df)
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


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_state_alpha_should_not_change_if_no_transition(gendf):
    df = gendf()

    engine = Engine(df, no_mp=True)
    engine.init_models(1)

    state_alpha_start = engine._models[0]['state_alpha']

    t_list = [b'row_assignment', b'row_alpha']
    engine.run(10, trans_kwargs={'transition_list': t_list})

    state_alpha_end = engine._models[0]['state_alpha']

    assert state_alpha_start == state_alpha_end


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_state_alpha_should_change_if_transition(gendf):
    df = gendf()

    engine = Engine(df, no_mp=True)
    engine.init_models(1)

    state_alpha_start = engine._models[0]['state_alpha']

    engine.run(10)

    state_alpha_end = engine._models[0]['state_alpha']

    assert state_alpha_start != state_alpha_end


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_view_alpha_should_not_change_if_no_transition(gendf):
    df = gendf()

    engine = Engine(df, no_mp=True)
    engine.init_models(1)

    view_alpha_start = engine._models[0]['view_alphas']

    t_list = [b'row_assignment', b'column_alpha']
    engine.run(10, trans_kwargs={'transition_list': t_list})

    view_alpha_end = engine._models[0]['view_alphas']

    assert view_alpha_start == view_alpha_end


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_view_alpha_should_change_if_transition(gendf):
    df = gendf()

    engine = Engine(df, no_mp=True)
    engine.init_models(1)

    view_alpha_start = engine._models[0]['view_alphas']

    engine.run(10)

    view_alpha_end = engine._models[0]['view_alphas']

    assert view_alpha_start != view_alpha_end


# save and load
# `````````````````````````````````````````````````````````````````````````````
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_save_smoke(gendf):
    df = gendf()

    engine = Engine(df)
    engine.init_models(5)

    with tempfile.NamedTemporaryFile('wb') as tf:
        engine.save(tf.name)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_load_smoke(gendf):
    df = gendf()

    engine = Engine(df)
    engine.init_models(5)

    with tempfile.NamedTemporaryFile('wb') as tf:
        engine.save(tf.name)
        Engine.load(tf.name)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_save_and_load_equivalence(gendf):
    df = gendf()

    engine = Engine(df)
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
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_single_col_single_datum(gendf):
    engine = gen_engine(gendf())

    data = 1.2
    col = 'x_1'
    p = engine.probability(data, [col])

    assert isinstance(p, (float, np.float64,))


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_multi_col_single_datum(gendf):
    engine = gen_engine(gendf())

    data = [1.2, 2.3]
    cols = ['x_1', 'x_4']
    p = engine.probability(data, cols)

    assert isinstance(p, (float, np.float64,))


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_single_col_multi_datum(gendf):
    engine = gen_engine(gendf())

    data = [[1.2], [0.2]]
    col = 'x_1'
    p = engine.probability(data, [col])

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_multi_col_multi_datum(gendf):
    engine = gen_engine(gendf())

    data = [[1.2, 'two'], [0.2, 'one']]
    cols = ['x_1', 'x_3']
    given = [('x_2', 0,), ('x_4', 0,)]
    p = engine.probability(data, cols, given=given)

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_single_col_single_datum_given(gendf):
    engine = gen_engine(gendf())

    data = 1.2
    col = 'x_1'
    given = [('x_2', 0,), ('x_4', 0,)]
    p = engine.probability(data, [col], given=given)

    assert isinstance(p, (float, np.float64,))


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_multi_col_single_datum_given(gendf):
    engine = gen_engine(gendf())

    data = [1.2, 2.3]
    cols = ['x_1', 'x_4']
    given = [('x_2', 0,), ('x_4', 0,)]
    p = engine.probability(data, cols, given=given)

    assert isinstance(p, (float, np.float64,))


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_single_col_multi_datum_given(gendf):
    engine = gen_engine(gendf())

    data = [[1.2], [0.2]]
    col = 'x_1'
    p = engine.probability(data, [col])

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_probability_multi_col_multi_datum_given(gendf):
    engine = gen_engine(gendf())

    data = [[1.2, 'two'], [0.2, 'one']]
    cols = ['x_1', 'x_3']
    given = [('x_2', 0,), ('x_4', 0,)]
    p = engine.probability(data, cols, given=given)

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)


# sample
# ````````````````````````````````````````````````````````````````````````````
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_sample_single_col_single_float(gendf):
    engine = gen_engine(gendf())

    cols = ['x_1']
    x = engine.sample(cols, n=1)

    assert isinstance(x, (float, np.float64,))


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_sample_single_col_single_str(gendf):
    engine = gen_engine(gendf())

    cols = ['x_3']
    x = engine.sample(cols, n=1)

    assert isinstance(x, str)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_sample_multi_col_single_mixed(gendf):
    engine = gen_engine(gendf())

    cols = ['x_1', 'x_3']
    x = engine.sample(cols, n=1)

    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)
    assert isinstance(x[0], (float, np.float64,))
    assert isinstance(x[1], str)


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_sample_multi_col_multi_mixed(gendf):
    engine = gen_engine(gendf())

    cols = ['x_1', 'x_3']
    x = engine.sample(cols, n=3)

    assert isinstance(x, np.ndarray)
    assert x.shape == (3, 2,)
    assert isinstance(x[0, 0], (float, np.float64,))
    assert isinstance(x[1, 0], (float, np.float64,))
    assert isinstance(x[2, 0], (float, np.float64,))

    assert isinstance(x[0, 1], str)
    assert isinstance(x[1, 1], str)
    assert isinstance(x[2, 1], str)


# suprisal
# ````````````````````````````````````````````````````````````````````````````
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_suprisal_default(gendf):
    df = gendf()
    engine = gen_engine(df)

    s = engine.suprisal('x_1')

    assert isinstance(s, pd.DataFrame)
    assert s.shape == (np.sum(pd.notnull(df['x_1'])), 2,)

    assert 'x_1' in s.columns
    assert 'surprisal' in s.columns


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
def test_suprisal_specify_rows(gendf):
    engine = gen_engine(gendf())

    s = engine.suprisal('x_1', rows=[2, 5, 11])

    assert isinstance(s, pd.DataFrame)
    assert s.shape == (3, 2,)

    assert 2 in s.index
    assert 5 in s.index
    assert 11 in s.index

    assert 'x_1' in s.columns
    assert 'surprisal' in s.columns


# impute
# ````````````````````````````````````````````````````````````````````````````
def test_impute_continuous_default():
    engine = gen_engine(smalldf_mssg())

    impdata = engine.impute('x_1')

    assert isinstance(impdata, pd.DataFrame)
    assert impdata.shape == (1, 2,)  # only one missing value


def test_impute_continuous_select_rows():
    engine = gen_engine(smalldf_mssg())

    impdata = engine.impute('x_1', rows=[1, 2, 3])

    assert isinstance(impdata, pd.DataFrame)
    assert impdata.shape == (3, 2,)


def test_impute_categorical():
    engine = gen_engine(smalldf_mssg())

    impdata = engine.impute('x_2')

    assert isinstance(impdata, pd.DataFrame)
    assert impdata.shape == (1, 2,)  # only one missing value


def test_impute_categorical_select_rows():
    engine = gen_engine(smalldf_mssg())

    impdata = engine.impute('x_2', rows=[1, 2, 3])

    assert isinstance(impdata, pd.DataFrame)
    assert impdata.shape == (3, 2,)

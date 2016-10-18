import os
import pytest
import tempfile
import pandas as pd
import numpy as np

from multiprocessing.pool import Pool

from baxcat.engine import Engine
from baxcat.metrics import SquaredError

from tempfile import NamedTemporaryFile


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


def unimodal_df():
    s1 = pd.Series(np.random.randn(200))
    s2 = pd.Series(np.random.randn(200))
    df = pd.concat([s1, s2], axis=1)
    df.columns = ['x_1', 'x_2']

    return df

def multimodal_df():
    s1 = pd.Series(np.hstack((np.random.randn(100)-3,
                              np.hstack(np.random.randn(100)+3))))
    s2 = pd.Series(np.hstack((np.random.randn(100)-3,
                              np.hstack(np.random.randn(100)+3))))
    df = pd.concat([s1, s2], axis=1)
    df.columns = ['x_1', 'x_2']

    return df


# generates engine w/o subsampling, but with rows in random order
def gen_engine_full(df):
    engine = Engine(df, n_models=4, use_mp=False)
    engine.init_models()
    engine.run(10)

    return engine

# generates engine w/ 0.5 subsample
def gen_engine_half(df):
    engine = Engine(df, n_models=4, use_mp=False)
    engine.init_models(subsample_size=0.5)
    engine.run(10)

    return engine


def gen_comp_engines(df, subsample_size):
    engine_full = Engine(df, n_models=8, use_mp=False)
    engine_full.init_models()
    engine_full.run(100)

    engine_mod = Engine(df, n_models=8, use_mp=False)
    engine_mod.init_models(subsample_size=subsample_size)
    engine_mod.run(100)

    return engine_full, engine_mod


# -- begin tests
@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
@pytest.mark.parametrize('init_func', [gen_engine_full, gen_engine_half])
def test_subsample_init(gendf, init_func):
    engine = init_func(gendf())

    assert engine._dtypes[0] == 'continuous'
    assert engine._dtypes[1] == 'categorical'
    assert engine._dtypes[2] == 'categorical'
    assert engine._dtypes[3] == 'continuous'


@pytest.mark.parametrize('gendf', [smalldf, smalldf_mssg])
@pytest.mark.parametrize('subsample_size', [None, 1.0, 0.5])
def test_save_and_load_equivalence(gendf, subsample_size):
    df = gendf()

    engine = Engine(df, n_models=5, use_mp=False)
    engine.init_models(subsample_size=subsample_size)

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


# -- Probability
@pytest.mark.parametrize('gendf', [unimodal_df, multimodal_df])
@pytest.mark.parametrize('subsample_size', [None, 1.0, 0.5])
def test_probability_equivalence(gendf, subsample_size):
    df = gendf()

    engine_full, engine_mod = gen_comp_engines(df, subsample_size)

    p_full = np.exp(engine_full.probability(-3, ['x_1']))
    p_mod = np.exp(engine_mod.probability(-3, ['x_1']))

    # probabilities should be w/in 0.05 abs error
    assert p_mod == pytest.approx(p_full, abs=0.05)

    p_full = np.exp(engine_full.probability(0.0, ['x_1']))
    p_mod = np.exp(engine_mod.probability(0.0, ['x_1']))

    assert p_mod == pytest.approx(p_full, abs=0.05)


# -- Suprisal
@pytest.mark.parametrize('gendf', [unimodal_df, multimodal_df])
@pytest.mark.parametrize('subsample_size', [None, 1.0, 0.5])
def test_surprisal_equivalence(gendf, subsample_size):
    df = gendf()

    engine_full, engine_mod = gen_comp_engines(df, subsample_size)

    s_full = engine_full.surprisal('x_1', [2]).iloc[0, 1]
    s_mod = engine_mod.surprisal('x_1', [2]).iloc[0, 1]

    # surprisal should be w/in 0.05 abs error
    assert s_mod == pytest.approx(s_full, rel=0.05)

    s_full = engine_full.surprisal('x_1', [150]).iloc[0, 1]
    s_mod = engine_mod.surprisal('x_1', [150]).iloc[0, 1]

    assert s_mod == pytest.approx(s_full, rel=0.05)

import pytest

import pandas as pd
import numpy as np

from baxcat.engine import Engine


@pytest.fixture()
def smalldf():

    s1 = pd.Series(np.random.rand(30))
    s2 = pd.Series([0.0, 1.0]*15)
    s3 = pd.Series(['one', 'two', 'three']*10)
    s4 = pd.Series(np.random.rand(30))

    return pd.concat([s1, s2, s3, s4], axis=1)


# test init
# `````````````````````````````````````````````````````````````````````````````
def test_engine_init_smoke_default(smalldf):
    df = pd.DataFrame(np.random.rand(30, 5))
    Engine(df)
    Engine(smalldf)


def test_engine_init_smoke_metadata(smalldf):
    metadata = dict()
    metadata[1] = {
        'dtype': 'categorical',
        'values': [-1, 0, 1, 99]}
    metadata[2] = {
        'dtype': 'categorical',
        'values': ['zero', 'one', 'two', 'three', 'four']}

    Engine(smalldf, metadata=metadata)
    Engine(smalldf, n_models=3, metadata=metadata)


# test run
# `````````````````````````````````````````````````````````````````````````````
def test_engine_run_smoke_default(smalldf):
    engine = Engine(smalldf)
    engine.run()
    engine.run(10)
    assert len(engine.metadata) == 1


def test_engine_run_smoke_multiple(smalldf):
    engine = Engine(smalldf, n_models=10)
    engine.run()
    engine.run(10)
    assert len(engine.metadata) == 10


# dependence probability
# `````````````````````````````````````````````````````````````````````````````
def test_dependence_probability():
    x = np.random.randn(30)

    s1 = pd.Series(x)
    s2 = pd.Series(x + 1.0)
    s3 = pd.Series(np.random.rand(30))

    df = pd.concat([s1, s2, s3], axis=1)
    df.columns = ['c0', 'c1', 'c2']

    engine = Engine(df, n_models=20)
    engine.run(50)
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

    engine = Engine(df, n_models=10)
    engine.run(5)

    depprob = engine.pairwise_func('dependence_probability')
    assert depprob.ix[0, 0] == 1.
    assert depprob.ix[1, 1] == 1.
    assert depprob.ix[2, 2] == 1.

    assert depprob.ix[0, 1] == depprob.ix[1, 0]
    assert depprob.ix[0, 2] == depprob.ix[2, 0]
    assert depprob.ix[1, 2] == depprob.ix[2, 1]

import pytest
import copy
import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal as mvn
from baxcat.miner import MInER


@pytest.fixture
def miner_df():
    s1 = pd.Series(['a', 'b', 'c', 'a', 'b']*10)
    s2 = pd.Series(np.random.rand(50))
    s3 = pd.Series(np.random.rand(50))

    df = pd.concat([s1, s2, s3], axis=1)
    df.columns = ['x_1', 'x_2', 'x_3']

    return df


def test_miner_init_smoke(miner_df):
    logcf = lambda row, x: mvn.logpdf(x, np.zeros(2), np.eye(len(x)))
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'])
    miner.init_models(2)
    assert hasattr(miner, '_logcf')
    assert hasattr(miner, '_miner_cols')
    assert hasattr(miner, '_miner_col_idxs')
    assert not hasattr(miner, 'combat_wombat')


def test_fit_smoke(miner_df):
    logcf = lambda row, x: mvn.logpdf(x, np.zeros(2), np.eye(len(x)))
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'])
    miner.init_models(2)
    miner.fit(1, 5)


def test_fit_changes_data_sometimes(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'])
    miner.init_models(2)
    miner.fit(1, 5)

    for i in range(miner._n_cols):
        assert(miner._df['x_1'].ix[i] == df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] != df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] != df['x_3'].ix[i])


def test_fit_changes_data_sometimes_one_col(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0
    miner = MInER(miner_df, logcf, ['x_2'])
    miner.init_models(2)
    miner.fit(1, 5)

    for i in range(miner._n_cols):
        assert(miner._df['x_1'].ix[i] == df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] != df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] == df['x_3'].ix[i])


def test_fit_changes_data_sometimes_one_col_categorical(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0
    miner = MInER(miner_df, logcf, ['x_1'])
    miner.init_models(2)
    miner.fit(1, 5)

    n_changed = 0
    for i in range(miner._n_cols):
        n_changed += miner._df['x_1'].ix[i] == df['x_1'].ix[i]
        assert(miner._df['x_2'].ix[i] == df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] == df['x_3'].ix[i])

    assert(n_changed > 0)


def test_fit_doesnt_change_data_sometimes(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: float('-Inf')
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'])
    miner.init_models(2)
    miner.fit(1, 5)

    for i in range(miner._n_cols):
        assert(miner._df['x_1'].ix[i] == df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] == df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] == df['x_3'].ix[i])

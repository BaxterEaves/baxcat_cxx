import pytest
from flaky import flaky
import copy
import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal as mvn
from scipy.stats import normaltest
from scipy.stats import norm

from baxcat.miner import MInER


@pytest.fixture
def miner_df():
    m = 15
    s1 = pd.Series(['a', 'b', 'c', 'a', 'b', 'c']*m)
    s2 = pd.Series(np.random.rand(6*m))
    s3 = pd.Series(np.random.rand(6*m))

    df = pd.concat([s1, s2, s3], axis=1)
    df.columns = ['x_1', 'x_2', 'x_3']

    return df


# ---
def test_miner_init_smoke(miner_df):
    logcf = lambda row, x: mvn.logpdf(x, np.zeros(2), np.eye(len(x)))
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'], n_models=2, use_mp=False)
    miner.init_models()
    assert hasattr(miner, '_logcf')
    assert hasattr(miner, '_miner_cols')
    assert hasattr(miner, '_miner_col_idxs')
    assert not hasattr(miner, 'combat_wombat')


@pytest.mark.slow
def test_fit_smoke(miner_df):
    logcf = lambda row, x: mvn.logpdf(x, np.zeros(2), np.eye(len(x)))
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(1, 5)

    assert(not np.any(np.isnan(miner._df['x_2'].values)))
    assert(not np.any(np.isnan(miner._df['x_3'].values)))


def test_fit_changes_data_sometimes(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(1, 5)

    assert(not np.any(np.isnan(miner._df['x_2'].values)))
    assert(not np.any(np.isnan(miner._df['x_3'].values)))

    for i in range(miner._n_cols):
        assert(miner._df['x_1'].ix[i] == df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] != df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] != df['x_3'].ix[i])


def test_fit_changes_data_sometimes_one_col(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0
    miner = MInER(miner_df, logcf, ['x_2'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(1, 5)

    assert(not np.any(np.isnan(miner._df['x_2'].values)))
    assert(not np.any(np.isnan(miner._df['x_3'].values)))

    for i in range(miner._n_cols):
        assert(miner._df['x_1'].ix[i] == df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] != df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] == df['x_3'].ix[i])


@flaky(max_runs=5, min_passes=1)
def test_fit_changes_data_sometimes_one_col_categorical(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0
    miner = MInER(miner_df, logcf, ['x_1'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(1, 5)

    assert(not np.any(np.isnan(miner._df['x_2'].values)))
    assert(not np.any(np.isnan(miner._df['x_3'].values)))

    n_changed = 0
    for i in range(miner._n_cols):
        n_changed += (miner._df['x_1'].ix[i] != df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] == df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] == df['x_3'].ix[i])

    assert(n_changed > 0)


def test_miner_changes_columns_differently(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: 0.0

    x_02 = df.loc[0, 'x_2']
    x_03 = df.loc[0, 'x_3']

    miner = MInER(miner_df, logcf, ['x_2', 'x_3'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(1, 5)

    assert miner._df.loc[0, 'x_2'] != x_02
    assert miner._df.loc[0, 'x_3'] != x_03
    for i in range(df.shape[0]):
        assert miner._df.loc[i, 'x_2'] != miner._df.loc[i, 'x_3']


def test_fit_doesnt_change_data_sometimes(miner_df):
    df = copy.deepcopy(miner_df)
    logcf = lambda row, x: float('-Inf')
    miner = MInER(miner_df, logcf, ['x_2', 'x_3'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(1, 5)

    assert(not np.any(np.isnan(miner._df['x_2'].values)))
    assert(not np.any(np.isnan(miner._df['x_3'].values)))

    for i in range(miner._n_cols):
        assert(miner._df['x_1'].ix[i] == df['x_1'].ix[i])
        assert(miner._df['x_2'].ix[i] == df['x_2'].ix[i])
        assert(miner._df['x_3'].ix[i] == df['x_3'].ix[i])


@pytest.mark.slow
@flaky(max_runs=5, min_passes=1)
def test_convert_uniform_column_to_normal(miner_df):
    logcf = lambda row, x: norm.logpdf(x[0], 0, 1)
    miner = MInER(miner_df, logcf, ['x_2'], n_models=2, use_mp=False)
    miner.init_models()
    miner.fit(20, 10)

    assert(not np.any(np.isnan(miner._df['x_2'].values)))
    assert(not np.any(np.isnan(miner._df['x_3'].values)))

    assert(normaltest(miner._df['x_2'])[1] > .05)
    assert(normaltest(miner._df['x_3'])[1] < .05)

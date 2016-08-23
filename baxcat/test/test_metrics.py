import pytest
import numpy as np
import pandas as pd

from baxcat import metrics


def gen_arrays():
    pred = np.array([0, 1, 1, 0, 1])
    obs = np.array([0, 1, 0, 1, 1])

    return obs, pred


def gen_series():
    pred = pd.Series([0, 1, 1, 0, 1])
    obs = pd.Series([0, 1, 0, 1, 1])

    return obs, pred


def gen_series_float():
    pred = pd.Series([0.1, 1.2, .8, .7, .2])
    # [0 1 1 0 0]
    # [0 1 0 0 0]
    obs = pd.Series([0.25, 1.1, .3, .6, .4])

    return obs, pred


# ---
@pytest.mark.parametrize('genfunc', [gen_arrays, gen_series])
def test_confmat(genfunc):

    obs, pred = genfunc()
    tp, tn, fp, fn = metrics.confmat(obs, pred)

    assert tp == 2
    assert tn == 1
    assert fp == 1
    assert fn == 1


# ---
def test_squared_error_binary():
    obs, pred = gen_series()
    se = metrics.SquaredError()

    assert se(obs, pred) == 2.


def test_squared_error_numeric():
    obs, pred = gen_series_float()
    se = metrics.SquaredError()

    assert abs(se(obs, pred) - 0.3325) < 10E-8


def test_relative_error():
    obs, pred = gen_series_float()
    re = metrics.RelativeError()

    assert abs(re(obs, pred) - 3.0242424242424244) < 10E-8


@pytest.mark.xfail(raises=ZeroDivisionError)
def test_relative_error_fails_on_data_with_zeros():
    obs, pred = gen_series_float()
    re = metrics.RelativeError()

    re(obs, pred)


def test_informedness_binary():
    obs, pred = gen_series()
    inf = metrics.Informedness()

    assert inf(obs, pred) == 2./3 + 1./2 - 1.


def test_informedness_threshold():
    obs, pred = gen_series_float()
    inf = metrics.Informedness(threshold=.8)

    assert inf(obs, pred) == 1./1 + 3./4 - 1


def test_markedness_binary():
    obs, pred = gen_series()
    mds = metrics.Markedness()

    assert mds(obs, pred) == 2./3. + 1./2. - 1


def test_markedness_threshold():
    obs, pred = gen_series_float()
    mds = metrics.Markedness(threshold=.8)

    assert mds(obs, pred) == 1./2 + 3./3. - 1


def test_corr_perfect_pos():
    obs = np.array([1, 2, 1, 2, 3], dtype=float)

    rho = metrics.Correlation()

    assert abs(rho(obs, obs) - 1.) < 10E-8


def test_corr_perfect_neg():
    obs = np.array([1, 2, 1, 2, 3], dtype=float)
    inf = -obs

    rho = metrics.Correlation()

    assert abs(rho(obs, inf) + 1.) < 10E-8

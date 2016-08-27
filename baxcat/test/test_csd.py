import pytest
import numpy as np

from baxcat.dist import csd
from pytest import approx


@pytest.fixture
def hypers():
    return {'dirichlet_alpha': 1.0}


@pytest.fixture
def suffstats():
    return {'n': 10, 'k': 3, '0': 1, '1': 4, '2': 5}


# --- values
def test_probability_values_1(hypers, suffstats):

    csd_value = csd.probability(0, suffstats, hypers)
    assert csd_value == approx(-1.87180217690159)


def test_probability_values_2(hypers, suffstats):
    csd_value = csd.probability(1, suffstats, hypers)
    assert csd_value == approx(-0.95551144502744)


def test_probability_values_3(hypers, suffstats):
    hypers['dirichlet_alpha'] = 2.5
    csd_value = csd.probability(0, suffstats, hypers)
    assert csd_value == approx(-1.6094379124341)


def test_probability_values_4(hypers, suffstats):
    hypers['dirichlet_alpha'] = .25
    suffstats['0'] = 2
    suffstats['1'] = 7
    suffstats['2'] = 13
    suffstats['n'] = 22

    csd_value = csd.probability(0, suffstats, hypers)
    assert csd_value == approx(-2.31363492918062)


# ---
# TODO: Coarse sample tests. In the future we should use Chi-square test.
def test_sample_proportions():
    n_samples = 10000

    hypers = {'dirichlet_alpha': 1.}
    suffstats = {'n': 2, 'k': 2, '0': 1, '1': 1}

    samples = csd.sample(suffstats, hypers, n=n_samples)
    bins = np.array(np.bincount(samples), dtype=float)
    assert len(bins) == 2
    bins /= n_samples
    assert abs(bins[0] - bins[1]) < .05

    suffstats = {'n': 10, 'k': 2, '1': 10}

    samples = csd.sample(suffstats, hypers, n=n_samples)
    bins = np.array(np.bincount(samples), dtype=float)
    assert len(bins) == 2
    bins /= n_samples

    assert abs(bins[0] - bins[1]) > .8

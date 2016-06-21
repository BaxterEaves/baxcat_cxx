import pytest
import numpy as np

from baxcat.dist import csd

ERROR_TOL = 10E-8


def assert_close(a, b):
    assert abs(a-b) < ERROR_TOL


def test_probability_values():
    hypers = {'alpha': 1.0}
    suffstats = {'n': 10, 'k': 3, '0': 1, '1': 4, '2': 5}

    msd_value = csd.probability(0, suffstats, hypers)
    assert_close(-1.87180217690159, msd_value)

    msd_value = csd.probability(1, suffstats, hypers)
    assert_close(-0.95551144502744, msd_value)

    hypers['alpha'] = 2.5
    msd_value = csd.probability(0, suffstats, hypers)
    assert_close(-1.6094379124341, msd_value)

    hypers['alpha'] = .25
    suffstats['0'] = 2
    suffstats['1'] = 7
    suffstats['2'] = 13
    suffstats['n'] = 22
    msd_value = csd.probability(0, suffstats, hypers)
    assert_close(-2.31363492918062, msd_value)


# TODO: Coarse sample tests. In the future we should use Chi-square test.
def test_sample_proportions():
    n_samples = 10000.

    hypers = {'alpha': 1.}
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

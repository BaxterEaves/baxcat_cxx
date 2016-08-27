import pytest

from baxcat.dist import nng
from pytest import approx


@pytest.fixture
def suffstats():
    return {'n': 4, 'sum_x': 10, 'sum_x_sq': 30}


@pytest.fixture
def hypers():
    return {'m': 2.1, 'r': 1.2, 's': 1.3, 'nu': 1.4}


# ---
def test_probability_values_1(suffstats, hypers):
    log_nng_pp = nng.probability(3, suffstats, hypers)
    assert log_nng_pp == approx(-1.28438638499611)


def test_probability_values_2(suffstats, hypers):
    log_nng_pp = nng.probability(-3, suffstats, hypers)
    assert log_nng_pp == approx(-6.1637698862186)

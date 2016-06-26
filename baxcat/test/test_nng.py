import pytest
import numpy as np

from baxcat.dist import nng

ERROR_TOL = 10E-8


def assert_close(a, b):
    assert abs(a-b) < ERROR_TOL


def test_probability_values():
    suffstats = {'n': 4, 'sum_x': 10, 'sum_x_sq': 30}
    hypers = {'m': 2.1, 'r': 1.2, 's': 1.3, 'nu': 1.4}

    log_nng_pp = nng.probability(3, suffstats, hypers)
    assert_close(log_nng_pp, -1.28438638499611)

    log_nng_pp = nng.probability(-3, suffstats, hypers)
    assert_close(log_nng_pp, -6.1637698862186)

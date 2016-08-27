import pytest
import numpy as np
import baxcat.utils.model_utils as mu

from pytest import approx


@pytest.fixture
def model():
    # this is not a full model, it's just enough to test model_utils.
    m = {
        'col_assignment': [0, 0],
        'dtypes': [b'categorical', b'continuous'],
        'view_alphas': [1.],
        'view_counts': [
            [5, 5]
        ],
        'col_suffstats': [
            [
                {'n': 5, 'k': 3, '0': 3, '1': 1, '2': 1},
                {'n': 5, 'k': 3, '0': 2, '2': 1, '2': 2}
            ], [
                {'n': 5, 'sum_x': 1.2936, 'sum_x_sq': 9.5996},
                {'n': 5, 'sum_x': 4.9492, 'sum_x_sq': 22.4901}
            ]
        ],
        'col_hypers': [
            {'dirichlet_alpha': 1.},
            {'m': 0, 'r': 1, 's': 1, 'nu': 1}
        ]
    }
    return m


def test_single_continuous_col_probability_values_1(model):
    # logp of 1 in column 1
    x = np.array([[1]], dtype=float)
    logp = mu.probability(x, [model], [1])

    assert logp == approx(-1.58025784097797)


def test_single_continuous_col_probability_values_2(model):
    x = np.array([[2]], dtype=float)
    logp = mu.probability(x, [model], [1])

    assert logp == approx(-2.01262102403666)


def test_single_categorical_col_probability_values_1(model):
    # logp of 1 in column 1
    x = np.array([[0]], dtype=int)
    logp = mu.probability(x, [model], [0])

    assert logp == approx(-0.848561284433976)


def test_single_categorical_col_probability_values_2(model):
    x = np.array([[2]], dtype=int)
    logp = mu.probability(x, [model], [0])

    assert logp == approx(-1.15710849534972)


def test_double_mixed_single_view_values(model):
    x = np.array([[0, 2.1]])
    logp = mu.probability(x, [model], [0, 1])

    assert logp == approx(-2.93134971834475)


def test_single_categorical_col_samples(model):
    n = 1000
    x = [int(y) for y in mu.sample([model], [0], n=n).flatten()]
    cts = np.bincount(x, minlength=3)

    assert cts[2] > cts[1]
    assert cts[0] > cts[2]

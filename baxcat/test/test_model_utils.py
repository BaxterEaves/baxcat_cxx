import pytest
import copy
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


# --- continuous impute confidence
def test_continuous_impute_confidence_identical():
    # this is not a full model, it's just enough to test confidence
    m = {
        'dtypes': [b'categorical', b'continuous'],
        'col_assignment': [0, 0],
        'row_assignments': [[0, 0, 0, 1, 1]],
        'col_suffstats': [
            [],
            [
                {},
                {'n': 5, 'sum_x': 4.9492, 'sum_x_sq': 22.4901}
            ]
        ],
        'col_hypers': [
            {'dirichlet_alpha': 1.},
            {'m': 0, 'r': 1, 's': 1, 'nu': 1}
        ]
    }
    row2idx = [dict((i, i) for i in range(5))]

    conf = mu._continuous_impute_conf([m]*3, row2idx*3, 1, 4)

    assert conf == pytest.approx(1.)


def test_continuous_impute_confidence_disjoint():
    # this is not a full model, it's just enough to test confidence
    m1 = {
        'dtypes': [b'categorical', b'continuous'],
        'col_assignment': [0, 0],
        'row_assignments': [[0, 0, 0, 1, 1]],
        'col_suffstats': [
            [],
            [
                {},
                {'n': 5,
                 'sum_x': 16.851501644605559,
                 'sum_x_sq': 57.47631023607466}
            ]
        ],
        'col_hypers': [
            {'dirichlet_alpha': 1.},
            {'m': 0, 'r': 1, 's': 1, 'nu': 1}
        ]
    }
    m2 = copy.deepcopy(m1)
    m2['col_suffstats'][1][1]['sum_x'] *= -1

    row2idx = [dict((i, i) for i in range(5))]

    conf = mu._continuous_impute_conf([m1, m2], row2idx*2, 1, 4)

    assert conf < .6


def test_continuous_impute_confidence_touching():
    # this is not a full model, it's just enough to test confidence
    m1 = {
        'dtypes': [b'categorical', b'continuous'],
        'col_assignment': [0, 0],
        'row_assignments': [[0, 0, 0, 1, 1]],
        'col_suffstats': [
            [],
            [
                {},
                {'n': 5,
                 'sum_x': 16.851501644605559,
                 'sum_x_sq': 57.47631023607466}
            ]
        ],
        'col_hypers': [
            {'dirichlet_alpha': 1.},
            {'m': 0, 'r': 1, 's': 1, 'nu': 1}
        ]
    }
    m2 = copy.deepcopy(m1)
    m2['col_suffstats'][1][1]['sum_x'] = 15.04834099047909
    m2['col_suffstats'][1][1]['sum_x_sq'] = 49.493741705631635

    row2idx = [dict((i, i) for i in range(5))]

    conf = mu._continuous_impute_conf([m1, m2], row2idx*2, 1, 4)

    assert conf > .5


# --- categorical impute confidence
def test_categorical_impute_confidence_identical():
    # this is not a full model, it's just enough to test confidence
    m = {
        'dtypes': [b'continuous', b'categorical'],
        'col_assignment': [0, 0],
        'row_assignments': [[0, 0, 0, 1, 1]],
        'col_suffstats': [
            [],
            [
                {},
                {'n': 5, 'k': 3, '0': 1, '1': 2, '2': 2}
            ]
        ],
        'col_hypers': [
            {'m': 0, 'r': 1, 's': 1, 'nu': 1},
            {'dirichlet_alpha': 1.}
        ]
    }
    row2idx = [dict((i, i) for i in range(5))]

    conf = mu._categorical_impute_conf([m]*3, row2idx*3, 1, 4)

    assert conf == pytest.approx(1.)


def test_categorical_impute_confidence_disjoint():
    # this is not a full model, it's just enough to test confidence
    m1 = {
        'dtypes': [b'continuous', b'categorical'],
        'col_assignment': [0, 0],
        'row_assignments': [[0, 0, 0, 1, 1]],
        'col_suffstats': [
            [],
            [
                {},
                {'n': 5, 'k': 3, '0': 0, '1': 1, '2': 4}
            ]
        ],
        'col_hypers': [
            {'m': 0, 'r': 1, 's': 1, 'nu': 1},
            {'dirichlet_alpha': 1.}
        ]
    }
    m2 = copy.deepcopy(m1)
    m2['col_suffstats'][1][1]['0'] = 4
    m2['col_suffstats'][1][1]['1'] = 1
    m2['col_suffstats'][1][1]['2'] = 0

    row2idx = [dict((i, i) for i in range(5))]

    conf = mu._categorical_impute_conf([m1, m2], row2idx*2, 1, 4)

    assert conf <= .5


def test_categorical_impute_confidence_touching():
    # this is not a full model, it's just enough to test confidence
    m1 = {
        'dtypes': [b'continuous', b'categorical'],
        'col_assignment': [0, 0],
        'row_assignments': [[0, 0, 0, 1, 1]],
        'col_suffstats': [
            [],
            [
                {},
                {'n': 5, 'k': 3, '0': 1, '1': 2, '2': 2}
            ]
        ],
        'col_hypers': [
            {'m': 0, 'r': 1, 's': 1, 'nu': 1},
            {'dirichlet_alpha': 1.}
        ]
    }
    m2 = copy.deepcopy(m1)
    m2['col_suffstats'][1][1]['0'] = 0
    m2['col_suffstats'][1][1]['1'] = 2
    m2['col_suffstats'][1][1]['2'] = 3

    row2idx = [dict((i, i) for i in range(5))]

    conf = mu._categorical_impute_conf([m1, m2], row2idx*2, 1, 4)

    assert conf > .5
    assert conf < .9

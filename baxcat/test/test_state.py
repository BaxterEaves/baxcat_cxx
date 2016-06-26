import pytest
import freeze
import numpy as np

from baxcat.state import BCState


@pytest.fixture
def blank():
    n_rows = 10
    n_cols = 4
    # column major
    X = np.random.rand(n_cols, n_rows)
    bcstate = BCState(X)
    return bcstate


@pytest.fixture
def gdat():
    n_rows = 10
    n_cols = 4
    # column major
    X = np.random.rand(n_cols, n_rows)
    datatypes_list = ['continuous']*n_cols
    distargs = [[0] for _ in range(n_cols)]
    return X, datatypes_list, distargs


# --- BEGIN SMOKE TESTS ---
def test_initalizes_base(gdat):
    X, datatypes_list, distargs = gdat
    bcstate = BCState(X, datatypes_list, distargs)
    assert bcstate is not None


def test_initalizes_no_args(gdat):
    X, _, _ = gdat
    bcstate = BCState(X)
    assert bcstate is not None


def test_transition(gdat):
    X, _, _ = gdat
    bcstate = BCState(X)
    bcstate.transition()


def test_get_metadata(gdat):
    X, _, _ = gdat
    bcstate = BCState(X)
    res = bcstate.get_metadata()
    assert res is not None


def test_predictive_draw(gdat):
    X, _, _ = gdat
    bcstate = BCState(X)
    res = bcstate.predictive_draw([[2, 2]], [[1, 3]], [4.1])
    res = bcstate.predictive_draw([[2, 2]])
    assert res is not None


def test_predictive_probability(gdat):
    X, _, _ = gdat
    bcstate = BCState(X)
    res = bcstate.predictive_probability([[2, 2]], [1.2], [[1, 3]], [0.0])
    res = bcstate.predictive_probability([[2, 2]], [1.2])
    assert res is not None


def test_conditioned_row_resample(gdat):
    X, _, _ = gdat
    bcstate = BCState(X)
    bcstate.conditioned_row_resample(2, lambda x: 0.0)
# --- END SMOKE TESTS ---


# ---
def test_predictive_draw_unconstrained_valid_output(blank):
    bcstate = blank
    res = bcstate.predictive_draw([[2, 2]])

    assert isinstance(res, list)
    assert len(res) == 1
    assert isinstance(res[0], list)
    assert len(res[0]) == 1
    assert isinstance(res[0][0], float)


def test_predictive_draw_unconstrained_m_draws_valid_output(blank):
    bcstate = blank
    m = 4
    res = bcstate.predictive_draw([[0, 0]], N=m)

    assert isinstance(res, list)
    assert len(res) == m
    for rm in res:
        assert isinstance(rm, list)
        assert len(rm) == 1
        assert isinstance(rm[0], float)


def test_predictive_draw_unconstrained_m_draws_j_query_valid_output(blank):
    bcstate = blank
    m = 4
    res = bcstate.predictive_draw([[0, 0], [1, 1]], N=m)

    assert isinstance(res, list)
    assert len(res) == m
    for rm in res:
        assert isinstance(rm, list)
        assert len(rm) == 2
        assert isinstance(rm[0], float)
        assert isinstance(rm[1], float)


# ---
def test_predictive_draw_constrained_invalid_input(blank):
    bcstate = blank
    with pytest.raises(ValueError):
        bcstate.predictive_draw([[2, 2], [1, 1]], [[0, 0]], [2., 3.])
    with pytest.raises(ValueError):
        bcstate.predictive_draw([[2, 2], [1, 1]], [[0, 0], [1, 1]], [2.])


def test_predictive_draw_constrained_valid_output(blank):
    bcstate = blank
    res = bcstate.predictive_draw([[2, 2]], [[0, 0]], [1.2])

    assert isinstance(res, list)
    assert len(res) == 1
    assert isinstance(res[0], list)
    assert len(res[0]) == 1
    assert isinstance(res[0][0], float)


def test_predictive_draw_constrained_m_draws_valid_output(blank):
    bcstate = blank
    m = 4
    res = bcstate.predictive_draw([[2, 2]], [[0, 0]], [1.2], N=m)

    assert isinstance(res, list)
    assert len(res) == m
    for rm in res:
        assert isinstance(rm, list)
        assert len(rm) == 1
        assert isinstance(rm[0], float)


def test_predictive_draw_constrained_m_draws_j_query_valid_output(blank):
    bcstate = blank
    m = 4
    res = bcstate.predictive_draw([[0, 0], [1, 1]], [[2, 2]], [1.2], N=m)

    assert isinstance(res, list)
    assert len(res) == m
    for rm in res:
        assert isinstance(rm, list)
        assert len(rm) == 2
        assert isinstance(rm[0], float)
        assert isinstance(rm[1], float)


# ---
def test_predictive_probability_unconstrained_invalid_input(blank):
    bcstate = blank
    with pytest.raises(ValueError):
        bcstate.predictive_probability([[2, 2], [1, 1]], [0.0])
    with pytest.raises(ValueError):
        bcstate.predictive_probability([[2, 2]], [0.0, 1.1])


def test_predictive_probability_unconstrained_valid_output(blank):
    bcstate = blank
    res = bcstate.predictive_probability([[2, 2]], [0.0])

    assert isinstance(res, list)
    assert len(res) == 1
    assert isinstance(res[0], float)


def test_predictive_probability_unconstrained_j_query_valid_output(blank):
    bcstate = blank
    res = bcstate.predictive_probability([[2, 2], [0, 1]], [0.0, 1.2])

    assert isinstance(res, list)
    assert len(res) == 2
    assert isinstance(res[0], float)
    assert isinstance(res[1], float)


# ---
def test_predictive_probability_constrained_invalid_input(blank):
    bcstate = blank
    with pytest.raises(ValueError):
        bcstate.predictive_probability([[2, 2]], [0.0], [[0, 0]], [1.2, 1.1])
    with pytest.raises(ValueError):
        bcstate.predictive_probability([[2, 2]], [0.0], [[0, 0], [1, 2]], [1.])


def test_predictive_probability_constrained_valid_output(blank):
    bcstate = blank
    res = bcstate.predictive_probability([[2, 2]], [0.0], [[0, 0]], [1.2])

    assert isinstance(res, list)
    assert len(res) == 1
    assert isinstance(res[0], float)


def test_predictive_probability_constrained_j_query_valid_output(blank):
    bcstate = blank
    res = bcstate.predictive_probability([[2, 2], [0, 1]], [0.0, 1.2],
                                         [[0, 0]], [1.2])

    assert isinstance(res, list)
    assert len(res) == 2
    assert isinstance(res[0], float)
    assert isinstance(res[1], float)


# ---
def test_get_metadata_valid_output(blank):
    bcstate = blank
    md = bcstate.get_metadata()

    assert isinstance(md, dict)

    assert isinstance(md['state_alpha'], float)

    assert isinstance(md['view_alphas'], list)
    assert isinstance(md['view_alphas'][0], float)

    assert isinstance(md['col_assignment'], list)
    assert isinstance(md['col_assignment'][0], int)

    assert isinstance(md['row_assignments'], list)
    for va in md['row_assignments']:
        assert isinstance(va, list)
        assert isinstance(va[0], int)

    assert isinstance(md['col_hypers'], list)
    for hp in md['col_hypers']:
        assert isinstance(hp, dict)

    assert isinstance(md['col_hypers'], list)
    for col_sf in md['col_suffstats']:
        assert isinstance(col_sf, list)
        for sf in col_sf:
            assert isinstance(sf, dict)


# ---
def test_transition_should_change_metadata(blank):
    bcstate = blank
    md_a = bcstate.get_metadata()
    hash_a = hash(freeze.freeze(md_a))

    md_a_cpy = bcstate.get_metadata()
    hash_a_cpy = hash(freeze.freeze(md_a_cpy))

    assert hash_a == hash_a_cpy

    bcstate.transition()

    md_b = bcstate.get_metadata()
    hash_b = hash(freeze.freeze(md_b))

    assert hash_a != hash_b


# ---
def test_conditioned_row_resample_should_change_metadata(blank):
    bcstate = blank

    # double exponential model
    logcf = lambda x: np.sum(-np.abs(x))

    md_a = bcstate.get_metadata()
    hash_a = hash(freeze.freeze(md_a))

    acr = bcstate.conditioned_row_resample(2, logcf, num_samples=1)
    while acr == 0:
        acr = bcstate.conditioned_row_resample(2, logcf, num_samples=1)

    assert acr > 0.0

    md_b = bcstate.get_metadata()
    hash_b = hash(freeze.freeze(md_b))

    assert hash_a != hash_b


def test_conditioned_row_resample_should_not_change_metadata_rejected(blank):
    bcstate = blank

    # double exponential model
    logcf = lambda x: float('-Inf')

    md_a = bcstate.get_metadata()
    hash_a = hash(freeze.freeze(md_a))

    acr = bcstate.conditioned_row_resample(2, logcf, num_samples=10)
    assert acr == 0.0

    md_b = bcstate.get_metadata()
    hash_b = hash(freeze.freeze(md_b))

    assert hash_a == hash_b

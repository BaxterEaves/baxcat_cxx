from baxcat.utils.test_utils import DataGenerator

import pytest


def test_sdg_init_single_categorical():
    sdg = DataGenerator(10, ['categorical'])

    assert sdg.df.shape == (10, 1,)
    assert sdg.df[0].dtype == 'int'
    assert len(sdg._params) == 1
    assert len(sdg._params[0]) == 2


def test_sdg_init_single_continuous():
    sdg = DataGenerator(10, ['continuous'])

    assert sdg.df.shape == (10, 1,)
    assert sdg.df[0].dtype == 'float'
    assert len(sdg._params) == 1
    assert len(sdg._params[0]) == 2


def test_sdg_init_dual_categorical():
    sdg = DataGenerator(10, ['categorical']*2)

    assert sdg.df.shape == (10, 2,)
    assert sdg.df[0].dtype == 'int'
    assert sdg.df[1].dtype == 'int'

    assert len(sdg._params) == 2
    assert len(sdg._params[0]) == 2
    assert len(sdg._params[1]) == 2


def test_sdg_init_dual_continuous():
    sdg = DataGenerator(10, ['continuous']*2)

    assert sdg.df.shape == (10, 2,)
    assert sdg.df[0].dtype == 'float'
    assert sdg.df[1].dtype == 'float'

    assert len(sdg._params) == 2
    assert len(sdg._params[0]) == 2
    assert len(sdg._params[1]) == 2


def test_sdg_init_dual_mixed():
    sdg = DataGenerator(10, ['continuous', 'categorical'])

    assert sdg.df.shape == (10, 2,)
    assert sdg.df[0].dtype == 'float'
    assert sdg.df[1].dtype == 'int'

    assert len(sdg._params) == 2
    assert len(sdg._params[0]) == 2
    assert len(sdg._params[1]) == 2


# ---
@pytest.mark.parametrize('dtype', ['categorical', 'continuous'])
def test_sdg_init_dual_mixed_one_view(dtype):
    sdg = DataGenerator(10, [dtype]*3, view_weights=1)

    assert sdg.df.shape == (10, 3,)
    assert min(sdg._colpart) == 0
    assert max(sdg._colpart) == 0


@pytest.mark.parametrize('dtype', ['categorical', 'continuous'])
def test_sdg_init_dual_mixed_two_view(dtype):
    sdg = DataGenerator(10, [dtype]*3, view_weights=2)

    assert sdg.df.shape == (10, 3,)
    assert min(sdg._colpart) == 0
    assert max(sdg._colpart) == 1


@pytest.mark.parametrize('dtype', ['categorical', 'continuous'])
def test_sdg_init_dual_mixed_three_view(dtype):
    sdg = DataGenerator(10, [dtype]*3, view_weights=3)

    assert sdg.df.shape == (10, 3,)
    assert min(sdg._colpart) == 0
    assert max(sdg._colpart) == 2

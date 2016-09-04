from baxcat.utils.test_utils import DataGenerator


def test_sdg_init_single_categorical():
    sdg = DataGenerator(10, ['categorical'])

    assert sdg.df.shape == (10, 1,)
    assert sdg.df[0].dtype == 'int'
    assert len(sdg._params) == 1
    assert len(sdg._params[0]) == 3


def test_sdg_init_single_continuous():
    sdg = DataGenerator(10, ['continuous'])

    assert sdg.df.shape == (10, 1,)
    assert sdg.df[0].dtype == 'float'
    assert len(sdg._params) == 1
    assert len(sdg._params[0]) == 3


def test_sdg_init_dual_categorical():
    sdg = DataGenerator(10, ['categorical']*2)

    assert sdg.df.shape == (10, 2,)
    assert sdg.df[0].dtype == 'int'
    assert sdg.df[1].dtype == 'int'

    assert len(sdg._params) == 2
    assert len(sdg._params[0]) == 3
    assert len(sdg._params[1]) == 3


def test_sdg_init_dual_continuous():
    sdg = DataGenerator(10, ['continuous']*2)

    assert sdg.df.shape == (10, 2,)
    assert sdg.df[0].dtype == 'float'
    assert sdg.df[1].dtype == 'float'

    assert len(sdg._params) == 2
    assert len(sdg._params[0]) == 3
    assert len(sdg._params[1]) == 3


def test_sdg_init_dual_mixed():
    sdg = DataGenerator(10, ['continuous', 'categorical'])

    assert sdg.df.shape == (10, 2,)
    assert sdg.df[0].dtype == 'float'
    assert sdg.df[1].dtype == 'int'

    assert len(sdg._params) == 2
    assert len(sdg._params[0]) == 3
    assert len(sdg._params[1]) == 3

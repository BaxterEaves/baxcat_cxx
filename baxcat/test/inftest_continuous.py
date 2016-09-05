from baxcat.engine import Engine
from baxcat.utils.test_utils import DataGenerator
from scipy.stats import ks_2samp

import numpy as np
import pytest


@pytest.mark.parametrize('n_rows', [50, 100, 250])
@pytest.mark.parametrize('n_cols', [1, 2, 4])
@pytest.mark.parametrize('n_cats', [1, 2, 3])
@pytest.mark.parametrize('cat_sep', [0.1, .3, .5, .8])
@pytest.mark.parametrize('n_models', [1, 4])
@pytest.mark.parametrize('n_iter', [200])
def test_ks(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg = DataGenerator(n_rows, ['continuous']*n_cols, cat_weights=n_cats,
                       cat_sep=cat_sep)
    engine = Engine(dg.df, use_mp=False)
    engine.init_models(n_models)
    engine.run(n_iter)

    cols = list(range(n_cols))

    df_sim = engine.sample(cols, n=n_rows)

    if n_cols == 1:
        x = dg.df[0].values
        y = np.array(df_sim[:, 0], dtype=float)
        d, p = ks_2samp(x, y)
        assert p > .05
    else:
        for col in cols:
            x = dg.df[col].values
            y = np.array(df_sim[:, col], dtype=float)
            d, p = ks_2samp(x, y)
            assert p > .05

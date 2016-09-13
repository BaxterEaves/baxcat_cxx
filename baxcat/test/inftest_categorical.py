import os
import numpy as np
import pytest

from scipy.stats import chisquare, pearsonr

from baxcat.engine import Engine
from baxcat.utils.test_utils import DataGenerator
from baxcat.utils.plot_utils import inftest_bar


DIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(DIR, 'result', 'csd')

if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)


def gen_data_and_engine(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg = DataGenerator(n_rows, ['categorical']*n_cols, cat_weights=n_cats,
                       cat_sep=cat_sep, seed=1337)
    col_md = {'dtype': 'categorical', 'values': [0, 1, 2, 3, 4]}
    md = dict((col, col_md,) for col in range(n_cols))
    engine = Engine(dg.df, metadata=md, use_mp=False)
    engine.init_models(n_models)
    engine.run(n_iter)

    return dg, engine


# ---
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('n_rows', [250, 500])
@pytest.mark.parametrize('n_cols', [1, 2, 4])
@pytest.mark.parametrize('n_cats', [1, 2, 3])
@pytest.mark.parametrize('cat_sep', [0.1, .3, .5, .8])
@pytest.mark.parametrize('n_models', [1, 4])
@pytest.mark.parametrize('n_iter', [200])
def test_ks(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg, engine = gen_data_and_engine(n_rows, n_cols, n_cats, cat_sep, n_models,
                                     n_iter)
    cols = list(range(n_cols))

    ttl_base = "CSD_X2-r%d-c%d-k%d-s%1.2f-m%d-i%d_" \
        % (n_rows, n_cols, n_cats, cat_sep, n_models, n_iter,)

    if n_cols == 1:
        y = sum(p['alpha'] for p in dg.params[0])/n_cats
        x = np.array(engine.sample([0], n=n_rows).flatten(), dtype=int)
        x = np.bincount(x, minlength=5)/n_rows
        d, p = chisquare(y, x)
        inftest_bar(y, x, ttl_base + 'COL-0', RESDIR)
        assert p > .05
    else:
        for col in cols:
            y = sum(p['alpha'] for p in dg.params[0])/n_cats
            x = np.array(engine.sample([0], n=n_rows).flatten(), dtype=int)
            x = np.bincount(x, minlength=5)/n_rows
            d, p = chisquare(y, x)

            inftest_bar(y, x, ttl_base + 'COL-%d' % col, RESDIR)
            assert p > .05


# ---
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('n_rows', [1000])
@pytest.mark.parametrize('n_cols', [1, 2])
@pytest.mark.parametrize('n_cats', [1, 2, 3])
@pytest.mark.parametrize('cat_sep', [0.1, .3, .5, .8])
@pytest.mark.parametrize('n_models', [4])
@pytest.mark.parametrize('n_iter', [200])
def test_likelihood(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg, engine = gen_data_and_engine(n_rows, n_cols, n_cats, cat_sep, n_models,
                                     n_iter)
    cols = list(range(n_cols))

    ttl_base = "CSD_LK-r%d-c%d-k%d-s%1.2f-m%d-i%d_" \
        % (n_rows, n_cols, n_cats, cat_sep, n_models, n_iter,)

    for col in cols:
        x = np.arange(5, dtype=int)

        l_dg = dg.log_likelihood(x, col)
        l_eng = engine.probability(x[:, np.newaxis], [col])

        r, _ = pearsonr(l_dg, l_eng)
        inftest_bar(np.exp(l_dg), np.exp(l_eng), ttl_base + 'COL-%d' % col,
                    RESDIR)

        assert abs(r-1) < .1

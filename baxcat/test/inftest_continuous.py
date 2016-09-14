import os
import numpy as np
import pytest

from scipy.stats import ks_2samp, pearsonr 

from baxcat.engine import Engine
from baxcat.utils.test_utils import DataGenerator
from baxcat.utils.plot_utils import inftest_plot, inftest_hist


DIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(DIR, 'result', 'nng')

if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)


def gen_data_and_engine(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg = DataGenerator(n_rows, ['continuous']*n_cols, cat_weights=n_cats,
                       cat_sep=cat_sep, seed=1337)
    engine = Engine(dg.df, use_mp=False)
    engine.init_models(n_models)
    engine.run(n_iter)

    return dg, engine


# ---
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('n_rows', [50, 100, 250])
@pytest.mark.parametrize('n_cols', [1, 2, 4])
@pytest.mark.parametrize('n_cats', [1, 2, 3])
@pytest.mark.parametrize('cat_sep', [0.1, .3, .5, .8])
@pytest.mark.parametrize('n_models', [1, 4])
@pytest.mark.parametrize('n_iter', [200])
def test_ks(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg, engine = gen_data_and_engine(n_rows, n_cols, n_cats, cat_sep, n_models,
                                     n_iter)
    cols = list(range(n_cols))

    df_sim = engine.sample(cols, n=n_rows)

    ttl_base = "NNG_KS-r%d-c%d-k%d-s%1.2f-m%d-i%d_" \
        % (n_rows, n_cols, n_cats, cat_sep, n_models, n_iter,)

    if n_cols == 1:
        x = dg.df[0].values
        y = np.array(df_sim[:, 0], dtype=float)
        d, p = ks_2samp(x, y)
        inftest_hist(y, x, ttl_base + 'COL-0', RESDIR)
        assert p > .05
    else:
        for col in cols:
            x = dg.df[col].values
            y = np.array(df_sim[:, col], dtype=float)
            d, p = ks_2samp(x, y)
            inftest_hist(y, x, ttl_base + 'COL-%d' % col, RESDIR)
            assert p > .05


# ---
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('n_rows', [250])
@pytest.mark.parametrize('n_cols', [1, 2])
@pytest.mark.parametrize('n_cats', [1, 2, 3])
@pytest.mark.parametrize('cat_sep', [0.1, .3, .5, .8])
@pytest.mark.parametrize('n_models', [4])
@pytest.mark.parametrize('n_iter', [200])
def test_likelihood(n_rows, n_cols, n_cats, cat_sep, n_models, n_iter):
    dg, engine = gen_data_and_engine(n_rows, n_cols, n_cats, cat_sep, n_models,
                                     n_iter)
    cols = list(range(n_cols))

    ttl_base = "NNG_LK-r%d-c%d-k%d-s%1.2f-m%d-i%d_" \
        % (n_rows, n_cols, n_cats, cat_sep, n_models, n_iter,)

    for col in cols:
        xmin = dg.df[col].min()
        xmax = dg.df[col].max()
        x = np.linspace(xmin, xmax, 200)

        l_dg = dg.log_likelihood(x, col)
        l_eng = engine.probability(x[:, np.newaxis], [col])

        r, _ = pearsonr(l_dg, l_eng)
        inftest_plot(x, l_dg, l_eng, ttl_base + 'COL-%d' % col, RESDIR)

        assert abs(r-1) < .05

import os
import pytest
import numpy as np
import pandas as pd

from scipy.stats import norm

from baxcat.utils.plot_utils import inftest_plot
from baxcat.engine import Engine


DIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(DIR, 'result', 'conditional')

if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)


@pytest.fixture
def df():
    n = 1000
    m = [[0., 1., 2.], [3., 4., 5.]]
    x = np.random.randint(2, size=n)
    y = np.random.randint(3, size=n)
    t = np.array([np.random.randn()*.5+m[xi][yi] for xi, yi in zip(x, y)])

    data = pd.concat([pd.Series(x), pd.Series(y), pd.Series(t)], axis=1)
    data.columns = ['x', 'y', 't']

    return data


def test_logp_scaling(df):
    engine = Engine(df)
    engine.init_models(8)
    engine.run(500)

    x = np.linspace(3, 7, 200)

    p_true = norm.pdf(x, loc=5., scale=.5)
    lp_baxcat = engine.probability(x[:, np.newaxis], ['t'],
                                   given=[('x', 1), ('y', 2)])

    inftest_plot(x, p_true, np.exp(lp_baxcat), 'p_t-xy', RESDIR)

    assert abs(max(p_true) - max(np.exp(lp_baxcat))) < .05

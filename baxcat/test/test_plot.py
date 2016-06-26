import pytest
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from baxcat.engine import Engine


DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_plots')


@pytest.fixture()
def smalldf():

    s1 = pd.Series(np.random.rand(30))
    s2 = pd.Series([0.0, 1.0]*15)
    s3 = pd.Series(['one', 'two', 'three']*10)
    s4 = pd.Series(np.random.rand(30))

    return pd.concat([s1, s2, s3, s4], axis=1)


# heatmap
# `````````````````````````````````````````````````````````````````````````````
def test_heatmap_dependence_probability():
    x = np.random.randn(30)
    s1 = pd.Series(x)
    s2 = pd.Series(x + 1.0)

    df = pd.concat([s1, s2]+[pd.Series(np.random.rand(30)) for _ in range(10)],
                   axis=1)
    df.columns = ['c_%d' % i for i in range(12)]

    engine = Engine(df)
    engine.init_models(10)
    engine.run(20)

    engine.heatmap('dependence_probability')

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    plt.savefig(os.path.join(DIR, 'heatmap-dp.png'), dpi=300)

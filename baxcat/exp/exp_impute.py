from baxcat.engine import Engine
from baxcat.misc import pflip

import numpy as np
import pandas as pd


def gen_mixture_data(n, mprop=.1):
    """ Generate 2-feature mixture data """
    x = np.zeros((n, 2,))
    weights = [0.3, 0.7]
    mu = [-1.0, 3.0]
    for i in range(n):
        k = pflip(weights)
        m = mu[k]
        x[i, :] = np.random.normal(m, size=2)

    df = pd.DataFrame(x)
    df.columns = ['x_1', 'x_2']

    return df


df = gen_mixture_data(200)

engine = Engine(df, n_models=8)
engine.init_models()
engine.run(200)

# find max value in x_1
amin = np.argmin(np.abs(df['x_1']-3.0))

resimp = engine.impute('x_1', [amin])

print(resimp)

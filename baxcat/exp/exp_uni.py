import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from baxcat.engine import Engine

x = np.hstack((
    np.random.randn(100) - 6,
    np.random.randn(100)*3,
    np.random.randn(100) + 6,))

s1 = pd.Series(x)
df = pd.DataFrame(s1, columns=['x'])

engine = Engine(df, n_models=8)
engine.run(100)
y = engine.sample('x', n=300)

sns.distplot(x, bins=30, label='original')
sns.distplot(y, bins=30, label='model')
plt.xlim([-10, 10])
plt.show()

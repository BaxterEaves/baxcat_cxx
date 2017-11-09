import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baxcat.engine import Engine


# Make a marker to phenotype model
# (-1, -1) -> m1, s
# (-1, 0) -> m2, s
# (-1, 1) -> m3, s
# etc...
def gen_phenotype_data(n_rows):
    data = []
    mus = np.array([[-.3, 0., 3.], [0., 3., 6.], [3., 6., 9.]])
    std = 1.
    for i in range(n_rows):
        a = np.random.randint(3)
        b = np.random.randint(3)
        mu = mus[a, b]
        x = np.random.randn()*std + mu

        data.append([x, a, b])

    return pd.DataFrame(data)

n_rows = 100
n_cols = 32

da = gen_phenotype_data(n_rows)
db = pd.DataFrame(np.random.randint(3, size=(n_rows, n_cols,)))
df = pd.concat([da, db], axis=1)

df.columns = ['T', 'A', 'B'] + ['x_%d' % i for i in range(n_cols)]

engine = Engine(df, n_models=32)
engine.init_models()
engine.run(100)

for col in df.columns:
    if col != 'T':
        print("1/H(%s|T) = %f" % (col, 1/engine.conditional_entropy(col, 'T')))

engine.heatmap('dependence_probability')
plt.show()

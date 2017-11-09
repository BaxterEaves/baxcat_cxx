"""
Make sure conditional probability works
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from baxcat.engine import Engine

sns.set_context("paper")

n = 200

s_a1 = pd.Series(np.zeros(n, dtype=int))
s_a2 = pd.Series(np.random.randn(n) - 2.)

s_b1 = pd.Series(np.ones(n, dtype=int))
s_b2 = pd.Series(np.random.randn(n) + 2.)


df = pd.concat([pd.concat([s_a1, s_a2], axis=1),
                pd.concat([s_b1, s_b2], axis=1)], axis=0)
assert df.shape == (2*n, 2,)

df.columns = ['label', 'x']


engine = Engine(df, n_models=8)
engine.init_models()
engine.run(200)

x = np.linspace(-6., 6., 200)[np.newaxis].T

p_01 = np.exp(engine.probability(x, ['x']))
p_0 = .5*np.exp(engine.probability(x, ['x'], given=[('label', 0,)]))
p_1 = .5*np.exp(engine.probability(x, ['x'], given=[('label', 1,)]))

plt.figure(figsize=(4, 4,))
plt.hist(df['x'], 31, histtype='stepfilled', color='#aaaaaa', edgecolor='None',
         normed=True)
plt.plot(x.flatten(), p_0, label='p(x|label=0)')
plt.plot(x.flatten(), p_1, label='p(x|label=1)')
plt.plot(x.flatten(), p_01, ls='--', label='p(x)')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend(loc=0)

plt.savefig('exp_condprob.png', dpi=300)

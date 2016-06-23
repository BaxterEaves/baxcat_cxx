"""
Mutual information experiment.

The mutual information should increase as the correlation between two columns
increases.
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from baxcat.engine import Engine

sns.set_context('paper')

n_times = 10
n_iter = 200
n = 50
m = 3
rhos = [.1, .25, .4, .5, .75, .9]

mi = np.zeros((n_times, len(rhos),))

for i, rho in enumerate(rhos):
    for t in range(n_times):
        print('Rho: %1.1f, time: %d' % (rho, t+1,))
        sigma = np.array([[1, rho], [rho, 1]])
        mu = np.zeros(2)
        x = np.random.multivariate_normal(mu, sigma, size=n)
        df = pd.DataFrame(x, columns=['x_1', 'x_2'])

        print('\tRunning')
        engine = Engine(df, n_models=1)
        engine.run(n_iter, trans_kwargs={'m': m})

        print('\tCalculating MI')
        mi[t, i] = engine.mutual_information('x_1', 'x_2', n_samples=500,
                                             normed=False)

rhoary = np.array(rhos)
true_mi = -.5*np.log(1-rhoary**2.)

plt.errorbar(rhos, y=np.mean(mi, axis=0), yerr=np.std(mi, axis=0),
             label='BaxCat')
plt.plot(rhos, true_mi, label='True')

plt.xlabel('rho')
plt.ylabel('Mutual Information')
plt.legend(loc=0)

plt.show()

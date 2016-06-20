import numpy as np


def pflip(p, normed=False, n=1):
    if normed:
        cs = np.cumsum(np.copy(p))
        if abs(cs[-1]-1.) > 10E-8:
            raise ValueError('You said the p vector was normed to 1. That '
                             'was a lie.')
    else:
        cs = np.cumsum(np.copy(p)/np.sum(p))

    rs = np.random.rand(n)

    if n == 1:
        draws = np.digitize([rs], cs)[0]
    else:
        draws = np.digitize(rs, cs)

    return draws

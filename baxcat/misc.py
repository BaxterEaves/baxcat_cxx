import numpy as np


def pflip(p, normed=False, n=1):
    if normed:
        cs = np.cumsum(np.copy(p))
        if abs(cs[-1]-1.) > 10E-8:
            raise ValueError('You said the p vector was normed to 1. That '
                             'was a lie.')
    else:
        cs = np.cumsum(np.copy(p)/np.sum(p))

    draws = np.zeros(n, dtype=int)
    for i in range(n):
        r = np.random.rand()
        for idx, w in enumerate(cs):
            if w > r:
                draws[i] = idx
                break

    if n == 1:
        return draws[0]
    else:
        return draws


def zpsave(data, filename):
    """ Save data as zipped pickle file. """
    raise NotImplementedError()


def zpload(filename):
    """ Load data from a zipped pickle file. """
    raise NotImplementedError()

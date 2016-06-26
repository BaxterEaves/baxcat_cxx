
from libcpp.vector cimport vector
from baxcat.misc import pflip

import numpy as np
from math import log


cdef _get_counts_from_suffstats(suffstats):
    cdef int k = int(suffstats['k'])
    cdef vector[size_t] counts = [suffstats.get(str(j), 0) for j in range(k)]
    return counts


def sample(suffstats, hypers, n=1):
    counts = _get_counts_from_suffstats(suffstats)
    alpha = hypers['dirichlet_alpha']

    ps = np.array(counts, dtype=float) + alpha
    ps /= np.sum(ps)

    assert len(ps) == len(counts)
    assert len(ps) == suffstats['k']

    return pflip(ps, normed=True, n=n)


def probability(x, suffstats, hypers):
    alpha = hypers['dirichlet_alpha']

    n = suffstats['n']
    k = suffstats['k']
    if suffstats['n'] == 0:
        return log(alpha) - log(alpha*suffstats['k'])
    else:
        ct = suffstats.get(str(int(x+.5)), 0.)
        return log(alpha+ct) - log(n+alpha*k)

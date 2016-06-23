
from libcpp.vector cimport vector
from baxcat.misc import pflip

import numpy as np


# FIXME: msd should be changed to csd when we get around to fixing the
# multinomial -> categorical bug.
cdef extern from "models/msd.hpp" namespace "baxcat::models":
    cdef cppclass MultinomialDirichlet[T]:  
        double logPredictiveProbability(T x, vector[T] counts,
                                        double alpha, double logZ)


def _get_counts_from_suffstats(suffstats):
    k = int(suffstats['k'])
    return [suffstats.get(str(j), 0) for j in range(k)]


# don't use cpp sampler, because it requires an rng
def sample(suffstats, hypers, n=1):
    counts = _get_counts_from_suffstats(suffstats)
    alpha = hypers['dirichlet_alpha']

    ps = np.array(counts, dtype=float) + alpha
    ps /= np.sum(ps)

    assert len(ps) == len(counts)
    assert len(ps) == suffstats['k']

    return pflip(ps, normed=True, n=n)


def probability(x, suffstats, hypers):
    cdef vector[size_t] counts = _get_counts_from_suffstats(suffstats)
    cdef double alpha = hypers['dirichlet_alpha']

    cdef MultinomialDirichlet[size_t] csd

    # XXX: logZ is a placeholder argument. Just give it wahetever.
    return csd.logPredictiveProbability(x, counts, alpha, 0)


from scipy.stats import t


cdef extern from "models/nng.hpp" namespace "baxcat::models":
    cdef struct NormalNormalGamma:  
        double logPredictiveProbability(double x, double n, double sum_x,
                                        double sum_x_sq, double m, double r,
                                        double s, double nu)


def _unwrap_suffstats(n=0., sum_x=0., sum_x_sq=0.):
    return n, sum_x, sum_x_sq


def _unwrap_hypers(m=0., r=1., s=1., nu=1.):
    return m, r, s, nu


def sample(suffstats, hypers):
    n, sum_x, sum_x_sq = _unwrap_suffstats(**suffstats)
    m, r, s, nu = _unwrap_hypers(**hypers)

    if n > 0:
        rn = r + n
        nun = nu + n
        mn = (r*m + sum_x)/(rn)
        sn = s + sum_x_sq + r*(m*m) - rn*(mn*mn)
    else:
        rn = r
        nun = nu
        mn = m
        sn = s

    scale = sn*(rn+1)/(nun*rn)

    return t.rvs(nun)*scale + mn
    

def probability(x, suffstats, hypers, seed=None):
    n, sum_x, sum_x_sq = _unwrap_suffstats(**suffstats)
    m, r, s, nu = _unwrap_hypers(**hypers)

    cdef NormalNormalGamma nng

    return nng.logPredictiveProbability(x, n, sum_x, sum_x_sq, m, r, s, nu)

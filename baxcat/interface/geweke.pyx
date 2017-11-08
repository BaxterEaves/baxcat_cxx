import os

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as cmap
from cython.operator import dereference

from scipy.stats import ks_2samp
from baxcat.utils import plot_utils as pu

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("notebook", font_scale=.8)


valid_dtypes = [
    "continuous",
    "categorical"]


cdef extern from "geweke_tester.hpp" namespace "baxcat":
    cdef cppclass GewekeTester:
        GewekeTester(size_t num_rows, size_t num_cols,
                     vector[string] datatypes, unsigned int seed, size_t m,
                     bool do_hypers, bool do_row_alpha, bool do_row_z,
                     bool do_col_z, bool ct_kernel) except +

        void run(size_t num_times, size_t num_posterior_chains, bool do_init)

        vector[cmap[string, vector[double]]] getForwardStats()
        vector[cmap[string, vector[double]]] getPosteriorStats()
        vector[double] getStateAlphaForward()
        vector[double] getStateAlphaPosterior()
        vector[size_t] getNumViewsForward()
        vector[size_t] getNumViewsPosterior()


cdef class Geweke:
    cdef GewekeTester *geweke
    cdef size_t n_cols
    cdef size_t n_rows
    cdef bool _do_col_z

    def __init__(self, n_rows, n_cols, dtypes, seed, m=1,
                 do_hypers=True, do_row_alpha=True, do_row_z=True,
                 do_col_z=True, ct_kernel=0):
        if len(dtypes) != n_cols:
            raise ValueError('Must be a dtype for each column')
        if not all(dt in valid_dtypes for dt in dtypes):
            raise ValueError('Invalid dtype in dtypes')
        dtypes = [dt.encode('utf-8') for dt in dtypes]

        self.n_cols = n_cols
        self.n_rows = n_rows
        self._do_col_z = do_col_z
        self.geweke = new GewekeTester(n_rows, n_cols, dtypes, seed, m,
                                       do_hypers, do_row_alpha, do_row_z,
                                       do_col_z, ct_kernel)

    def run(self, n_samples, n_chains, lag):
        self.geweke.run(n_samples, n_chains, lag)

    def output(self, resdir):
        stats_f = list(self.geweke.getForwardStats())
        stats_p = list(self.geweke.getPosteriorStats())

        fnames = ['col_%i' % i for i in range(self.n_cols)]

        if self._do_col_z:
            alpha_f = self.geweke.getStateAlphaForward()
            alpha_p = self.geweke.getStateAlphaPosterior()
            views_f = self.geweke.getNumViewsForward()
            views_p = self.geweke.getNumViewsPosterior()

            stats_f.append({'n views': views_f, 'state alpha': alpha_f})
            stats_p.append({'n views': views_p, 'state alpha': alpha_p})

            assert len(stats_f) == self.n_cols + 1
            assert len(stats_p) == self.n_cols + 1

            fnames.append('state')

        n_failures = 0
        for i, (stats_fi, stats_pi,) in enumerate(zip(stats_f, stats_p)):
            print("%s statistics:" % fnames[i])
            n_stats = len(stats_fi.keys())
            fig = plt.figure(figsize=(n_stats*2, 6))
            for j, key in enumerate(stats_fi.keys()):
                stat_fi = stats_fi[key]
                stat_pi = stats_pi[key]

                d, p = ks_2samp(stat_fi, stat_pi)
                if p > .05:
                    pstxt = "PASS"
                else:
                    n_failures += 1
                    pstxt = "FAIL"

                print("\tKS-test on %s = %f (p=%f) [%s]" % (key, d, p, pstxt,))
    
                ax1 = plt.subplot(3, n_stats, j+1)
                ax1.hist(stat_fi, normed=True)
                ax1.set_xlabel('%s (forward)' % (key,))
                ax1.set_title('%s [%s]' % (key, pstxt))

                ax2 = plt.subplot(3, n_stats, n_stats+j+1)
                ax2.hist(stat_pi, normed=True)
                ax2.set_xlabel('%s (posterior)' % (key,))
                ax2.set_xlim(ax1.get_xlim())

                ax3 = plt.subplot(3, n_stats, 2*n_stats+j+1)
                pu.pp_plot(stat_fi, stat_pi, ax=ax3)
                ax3.set_xlabel('forward')
                ax3.set_ylabel('posterior')

            fig.tight_layout()
            plt.savefig(os.path.join(resdir, '%s.png' % fnames[i]), dpi=150)

        if n_failures > 0:
            print("%d tests failed." % (n_failures,))
            return 1
        else:
            print("No failures detected.")
            return 0

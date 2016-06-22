from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as cmap
from cython.operator import dereference

from baxcat.utils import validation
from math import log
from random import random as rand
import numpy as np

import numpy
import time

valid_datatypes = [
    "continuous",
    "categorical"]


valid_transitions = [
    "row_assignment",
    "column_assignment",
    "row_alpha",
    "column_alpha",
    "column_hypers"]


cdef extern from "state.hpp" namespace "baxcat":
    cdef cppclass State:
        State(vector[vector[double]] X,
              vector[string] dtypes,
              vector[vector[double]] distargs,
              size_t rng_seed) except +

        State(vector[vector[double]] X,
              vector[string] dtypes,
              vector[vector[double]] distargs,
              size_t rng_seed,
              vector[size_t] Zv,
              vector[vector[size_t]] Zrcv,
              vector[cmap[string, double]] hyper_maps) except +

        void transition(vector[string] transition_list,
                        vector[size_t] which_rows,
                        vector[size_t] which_cols,
                        size_t which_kernel,
                        size_t N,
                        size_t m)

        # getters
        vector[size_t] getColumnAssignment()
        vector[vector[size_t]] getRowAssignments()
        vector[double]getDataRow(size_t row_index)
        vector[cmap[string, double]] getColumnHypers()
        vector[double] getViewCRPAlphas()
        double getStateCRPAlpha()
        size_t getNumViews()
        vector[vector[size_t]] getViewCounts()
        vector[vector[cmap[string, double]]] getSuffstats()
        double logScore();

        # setters
        void setHyperConfig(size_t column_index,
                            vector[double] hyperprior_config)
        void setHypers(size_t column_index, cmap[string, double] hypers_map)
        void setHypers(size_t column_index, vector[double] hypers_vec)

        # append and pop row
        void appendRow(vector[double] data_row, bool assign_to_max_p_cluster)
        void replaceRowData(size_t row_index, vector[double] new_row_data)
        void popRow()

        # predictive functions
        vector[double] predictiveLogp(vector[vector[size_t]] query_indices,
                                      vector[double] query_values,
                                      vector[vector[size_t]] constraint_indices,
                                      vector[double] constraint_values)

        vector[vector[double]] predictiveDraw(
                vector[vector[size_t]] query_indices,
                vector[vector[size_t]] constraint_indices,
                vector[double] constraint_values,
                size_t N)


cdef emptyvecvec(intype):
    cdef vector[vector[double]] ret_double
    cdef vector[vector[size_t]] ret_size_t
    if intype == 'double':
        return ret_double
    elif intype == 'size_t':
        return ret_size_t
    else:
        raise ValueError


cdef emptyvec(intype):
    cdef vector[double] ret_double
    cdef vector[size_t] ret_size_t
    if intype == 'double':
        return ret_double
    elif intype == 'size_t':
        return ret_size_t
    else:
        raise ValueError


def dictstr_dec(d):
    return dict([(k.decode('utf-8'), v) for k, v in d.items()])

def dictstr_enc(d):
    return dict([(k.encode(), v) for k, v in d.items()])


cdef class BCState:
    cdef State *statePtr
    cdef size_t n_rows
    cdef size_t n_cols
    cdef vector[string] datatypes

    def __cinit__(self, X, dtypes=None, distargs=None, col_hypers=None,
                  Zv=None, Zrcv=None, n_grid=31, seed=None):
        # data is column major (the rows in X become the crosscat columns)
        self.n_cols, self.n_rows = X.shape
        if seed is None or seed < 0:
            seed = int(time.time())
        if dtypes is None:
            dtypes = ['continuous']*self.n_cols
        if distargs is None:
            # 'empty'
            distargs = np.zeros((self.n_cols, 1))
       
        if len(dtypes) != self.n_cols:
            raise ValueError("Should be a dtype for each column.")
        if len(distargs) != self.n_cols:
            raise ValueError("Should be a distarg for each column.")

        dtl = [bytes(st, 'ascii') for st in dtypes]
        self.datatypes = dtl 

        if all(m == None for m in[col_hypers, Zv, Zrcv]):
            self.statePtr = new State(X, dtl, distargs, seed)
        elif not any(m == None for m in [col_hypers, Zv, Zrcv]):
            col_hypers = [dictstr_enc(hyper) for hyper in col_hypers]
            self.statePtr = new State(X, dtl, distargs, seed, Zv, Zrcv,
                                      col_hypers)
        else:
            raise ValueError('You have specified some, but not all of, Zv, '
                             'Zrcv, and col_hypers. There is currently no '
                             'initializer to handle this.')


    def __dealloc__(self):
        del self.statePtr

    def log_score(self):
        """ Returns the log score of the state. Runs in O(rows*cols). """
        return self.statePtr.logScore()


    def transition(self, transition_list=(), which_rows=(), which_cols=(),
                   which_kernel=0, N=1, m=1):
        # TODO: validate input
        self.statePtr.transition(transition_list, which_rows, which_cols,
                                 which_kernel, N, m)

    def predictive_probability(self, query_indices, query_values,
                               constraint_indices=None,
                               constraint_values=None):
        """
        Get the predictive probability
        """
        validation.validate_index_list(query_indices, list_type='query')
        if not isinstance(query_values, list):
            raise TypeError('query_values must be a list')
        if len(query_indices) != len(query_values):
            raise ValueError('query_indices and query_values must have the '
                             'same number of values')

        if constraint_indices is None:
            constraint_indices = emptyvecvec('size_t')
        else:
            validation.validate_index_list(constraint_indices,
                                           list_type='constraint')

        if constraint_values is None:
            constraint_values = emptyvec('double')
        else:
            if not isinstance(constraint_values, list):
                raise TypeError('constraint_values must be a list')
            if len(constraint_values) != len(constraint_indices):
                raise ValueError('constraint_indices and constraint_values '
                                 'must be the same length')

        return self.statePtr.predictiveLogp(query_indices, query_values,
                                            constraint_indices,
                                            constraint_values)

    def predictive_draw(self, query_indices, constraint_indices=None,
                        constraint_values=None, N=1):
        """
        Predictive draw from state.
        """
        # FIXME: no constraints doesn't work.
        validation.validate_index_list(query_indices, list_type='query')

        if constraint_indices is None:
            constraint_indices = emptyvecvec('size_t')
        else:
            validation.validate_index_list(constraint_indices,
                                           list_type='constraint')

        if constraint_values is None:
            constraint_values = emptyvec('double')
        else:
            if not isinstance(constraint_values, list):
                raise TypeError('constraint_values must be a list')
            if len(constraint_values) != len(constraint_indices):
                raise ValueError('constraint_indices and constraint_values '
                                 'must be the same length')

        return self.statePtr.predictiveDraw(query_indices, constraint_indices,
                                            constraint_values, N)

    def get_metadata(self):
        metadata = dict()

        metadata['dtypes'] = self.datatypes
        metadata['col_assignment'] = self.statePtr.getColumnAssignment()
        metadata['row_assignments'] = self.statePtr.getRowAssignments()
        # metadata['hyperprior_configs'] = []  # what is this for? 
        hypers = self.statePtr.getColumnHypers()
        metadata['col_hypers'] = [dictstr_dec(hp) for hp in hypers]
        metadata['state_alpha'] = self.statePtr.getStateCRPAlpha()
        metadata['view_alphas'] = self.statePtr.getViewCRPAlphas()
        suffstats = self.statePtr.getSuffstats()
        sfsts = []
        for col_sfst in suffstats:
            sfsts.append([dictstr_dec(sfst) for sfst in col_sfst])
        metadata['col_suffstats'] = sfsts
        metadata['view_counts'] = self.statePtr.getViewCounts() 

        # FIXME: add view_counts field
        return metadata

    def conditioned_row_resample(self, row_index, logcf, num_samples=10):
        # TODO: verify that this example works
        """
        Resamples a row in state conditioned on some external log probability
        function.

        Parameters
        ----------
        row_index : int
            row index of the cell to resample
        logcf : function
            function that returns the log probability of the row given some
            external criterion.
        num_samples : int, optional
            number of Metropolis-Hastings steps to do

        Examples
        --------
        condition resample given n-dimensional multivariate normal distribution
        >>> import numpy as np
        >>> from scipy.stats import multivariate_normal as mvn
        >>> # generate random data and state
        >>> X = np.random.randn(4,100).tolist()  # baxcat is column major
        >>> state = baxcat.State(X, ['continuous']*4)
        >>> # generate evaluation function
        >>> mu = np.zeros(4)   # normal mean
        >>> sigma = np.eye(4)  # normal standard deviation
        >>> logcf = lambda row_data : mvn.logpdf(row_data, mu, sigma)
        >>> # resample row 5 (50 Metropolis-Hastings steps)
        >>> state.conditioned_row_resample(5, logcf, num_samples=50)
        """
        if not isinstance(row_index, int):
            raise ValueError('row_index must be an integer index.')
        if not callable(logcf):
            raise ValueError('logcf must be a function')

        def get_random_row(row_index, num_cols):
            # FIXME: this is not quite right, because the existing suffstats
            # in the row must be removed before the new row is sampled (and
            # then replaced afterward).
            query_indices = [[row_index, col_index] for col_index in 
                              range(num_cols)]
            return self.predictive_draw(query_indices)[0]

        def row_predictive_probability(row_index, Y):
            logp = 0.0
            query_indices = []
            query_values = []
            for col_index, y in enumerate(Y):
                query_indices.append([row_index, col_index])
                query_values.append(y)
            logps = self.predictive_probability(query_indices, query_values)
            return np.sum(logps)

        acr = 0
        # get current data from the row
        y = self.statePtr.getDataRow(row_index)
        lc = logcf(y)

        if not isinstance(lc, float):
            raise ValueError('logcf should return a single float (log P)')

        l = row_predictive_probability(row_index, y) + lc
        for _ in range(num_samples):
            # get a sample from the row
            yp = get_random_row(row_index, self.n_cols)
            print(yp)
            lp = row_predictive_probability(row_index, yp) + logcf(yp)

            # calcualte acceptance probability
            if log(rand()) < lp-l:
                acr += 1
                l = lp
                y = yp
                self.statePtr.replaceRowData(row_index, y)

        return acr/float(num_samples)

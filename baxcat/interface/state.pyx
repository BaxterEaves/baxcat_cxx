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
              double state_alpha,
              vector[double] view_alpha,
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

        vector[double] getViewLogps();
        vector[double] getFeatureLogps();
        vector[vector[double]] getClusterLogps();
        vector[vector[double]] getRowLogps();

        # setters
        void setHyperConfig(size_t column_index,
                            vector[double] hyperprior_config)
        void setHypers(size_t column_index, cmap[string, double] hypers_map)
        void setHypers(size_t column_index, vector[double] hypers_vec)

        # append and pop row
        void appendRow(vector[double] data_row, bool assign_to_max_p_cluster)
        void replaceDatum(size_t row_idx, size_t col_idx, double new_datum)
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
                  Zv=None, Zrcv=None, state_alpha=-1, view_alphas=None,
                  n_grid=31, seed=None):
        # data is column major (the rows in X become the crosscat columns)
        self.n_cols, self.n_rows = X.shape
        if seed is None or seed < 0:
            seed = int(time.time())

        if dtypes is None:
            dtypes = ['continuous']*self.n_cols

        if distargs is None:
            # 'empty'
            distargs = np.zeros((self.n_cols, 1))

        if view_alphas is None:
            view_alphas = []
       
        if len(dtypes) != self.n_cols:
            raise ValueError("Should be a dtype for each column.")
        if len(distargs) != self.n_cols:
            raise ValueError("Should be a distarg for each column.")

        dtl = [bytes(st, 'ascii') for st in dtypes]
        self.datatypes = dtl 

        if all(m == None for m in[col_hypers, Zv, Zrcv]):
            self.statePtr = new State(X, dtl, distargs, seed)
        elif all(m is not None for m in [col_hypers, Zv, Zrcv]):
            col_hypers = [dictstr_enc(hyper) for hyper in col_hypers]
            self.statePtr = new State(X, dtl, distargs, seed, Zv, Zrcv,
                                      state_alpha, view_alphas, col_hypers)
        elif (col_hypers is None) and (Zv is not None) and (Zrcv is not None):
            self.statePtr = new State(X, dtl, distargs, seed, Zv, Zrcv,
                                      state_alpha, view_alphas, [])
        else:
            raise ValueError('No initializer for this variable set.')


    def __dealloc__(self):
        del self.statePtr

    def log_score(self):
        """ Returns the log score of the state. Runs in O(rows*cols). """
        return self.statePtr.logScore()

    @property
    def n_views(self):
        return self.statePtr.getNumViews()

    def transition(self, transition_list=(), which_rows=(), which_cols=(),
                   which_kernel=0, N=1, m=1):
        # TODO: validate input
        self.statePtr.transition(transition_list, which_rows, which_cols,
                                 which_kernel, N, m)

    def get_logps(self):
        logps = {}
        logps['view_logps'] = self.statePtr.getViewLogps()
        logps['column_logps'] = self.statePtr.getFeatureLogps()
        logps['row_logps'] = self.statePtr.getRowLogps()
        logps['cluster_logps'] = self.statePtr.getClusterLogps()
        return logps


    def predictive_logp(self, query_indices, query_values,
                        constraint_indices=None, constraint_values=None):
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

        return metadata

    def replace_data(self, idxs, data):
        """ Replace existing data with new data. 

        Parameters
        ----------
        idxs : list of (row, col,) tuples
            List of cells to update
        data : list
            List of data. Should have an entry for each entry in `idxs`.
        """
        if len(idxs) != len(data):
            raise ValueError('data and idxs should have the same length')

        for (row_idx, col_idx,), x in zip(idxs, data):
            self.statePtr.replaceDatum(row_idx, col_idx, x)

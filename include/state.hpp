
// BaxCat: an extensible cross-catigorization engine.
// Copyright (C) 2014 Baxter Eaves
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License
// (LICENSE.txt) along with this program. If not, see 
// <http://www.gnu.org/licenses/>.
//
// You may contact the mantainers of this software via github
// <https://github.com/BaxterEaves/baxcat_cxx>.

#ifndef baxcat_cxx_state_guard
#define baxcat_cxx_state_guard

#include <map>
#include <string>
#include <memory>
#include <vector>
#include <typeinfo>
#include "omp.h"

#include "prng.hpp"
#include "view.hpp"
#include "feature.hpp"
#include "helpers/feature_builder.hpp"
#include "helpers/state_helper.hpp"
#include "distributions/gamma.hpp"
#include "helpers/constants.hpp"
#include "samplers/slice.hpp"


// State class
// ```````````````````````````````````````````````````````````````````````````
namespace baxcat {

class State{
public:

    State(){};

    // Gewke init mode
    State(size_t num_rows, std::vector<std::string> datatypes,
          std::vector<std::vector<double>> distargs,
          bool fix_hypers,
          bool fix_row_alpha, bool fix_col_alpha,
          bool fix_row_z, bool fix_col_z);

    // init from prior
    // X is the table of data. X[f] is the data for feature X. Is cast to
    // proper type
    // datatypes[f] is the datatype for feature feature
    // n_grid is the number of bins for hyperparameters
    // rng_seed seeds the rng starting state
    State(std::vector<std::vector<double>> X,
        std::vector<std::string> datatypes,
        std::vector<std::vector<double>> distargs,
        unsigned int rng_seed);

    // init with a set partition
    // Zv[f] is the view to which feature f belongs
    // Zrcv[v][r] is the category to which row r of the features in view v are
    // assigned
    State(std::vector<std::vector<double>> X,
          std::vector<std::string> datatypes,
          std::vector<std::vector<double>> distargs,
          unsigned int rng_seed,
          std::vector<size_t> Zv,
          std::vector<std::vector<size_t>> Zrcv,
          std::vector<std::map<std::string, double>> hyper_maps);

    // do transitions.
    void transition(std::vector<std::string> which_transitions,
        std::vector<size_t> which_rows, std::vector<size_t> which_cols,
        size_t which_kernel, int N, size_t m=1);

    // getters
    std::vector<std::vector<double>> getDataTable() const;
    std::vector<double> getDataRow(size_t row_index) const;
    std::vector<size_t> getColumnAssignment() const;
    std::vector<std::vector<size_t>> getRowAssignments() const;
    std::vector<std::map<std::string, double>> getColumnHypers() const;
    std::vector<double> getViewCRPAlphas() const;
    double getStateCRPAlpha() const;
    size_t getNumViews() const;
    std::vector<std::vector<std::map<std::string, double>>> getSuffstats() const;

    // setters
    void setHyperConfig(size_t column_index, std::vector<double>
                        hyperprior_config);
    void setHypers(size_t column_index,
                   std::map<std::string, double> hypers_map);
    void setHypers(size_t column_index, std::vector<double> hypers_vec);

    // predictive_logp
    // returns the logp of the values in query_values being in corresponding
    // indices in query_indices given that the values in constraint_values are
    // in constraint_indices
    std::vector<double> predictiveLogp(
        std::vector<std::vector<size_t>> query_indices,
        std::vector<double> query_values,
        std::vector<std::vector<size_t>> constraint_indices,
        std::vector<double> constraint_values);

    // append data to last row. assign_to_p_max_row specifies whether the row 
    // is assigned to the category with the max probability or is assigned
    // probabilistically
    void appendRow(std::vector<double> data_row, bool assign_to_max_p_cluster);
    void replaceSliceData(std::vector<size_t> row_range,
                          std::vector<size_t> col_range,
                          std::vector<std::vector<double>> new_data);
    void replaceRowData(size_t row_index, std::vector<double> new_row_data);
    void popRow();

    // predictive_draw
    // does N draws from query indices given constraint_values returns an N by
    // query_indices.size() vector of vectors
    std::vector<std::vector<double>> predictiveDraw(
        std::vector<std::vector<size_t>> query_indices,
        std::vector<std::vector<size_t>> constraint_indices,
        std::vector<double> constraint_values,
        size_t N);

    // for geweke
    // clear suffstats and data
    void __geweke_clear();
    // resample one row
    void __geweke_resampleRows();
    // resample all rows
    void __geweke_resampleRow(size_t which_row);
    std::vector<double> __geweke_pullDataColumn(size_t column_index) const;
    // void __geweke_initHypers();

    // for debugging
    int checkPartitions();

    // TODO: implement these features
    // void setColumnHypers(std::map<string, double> hypers, size_t which_col);
    // void appendFeature(std::vector<double> data_column, std::string datatype);

private:

    // METHODS
    void __doTransition(baxcat::transition_type t,
                        std::vector<size_t> which_rows,
                        std::vector<size_t> which_cols, size_t which_kernel, 
                        size_t m=1);

    // Transition methods
    // transition state alpha --- alpha over columns
    void __transitionStateCRPAlpha();
    // transition view alpha --- alpha over rows in views
    void __transitionViewCRPAlphas();
    // transition hyperparameters for columns in which_cols
    // if which_cols is empty, all columns, in shuffled order are transistioned
    void __transitionColumnHypers(std::vector<size_t> which_columns);
    // transition the assignment of columns to views
    // if which_cols is empty, all columns, in shuffled order are transistioned
    void __transitionColumnAssignment(std::vector<size_t> which_columns, 
                                      size_t which_kernel, size_t m);
    // transition the assignment of rows in views to categories
    // if which_rows is empty, all rows, in shuffled order are transistioned
    // shuffling is handled by the view
    void __transitionRowAssignments(std::vector<size_t> which_rows);

    // Column transition kernels
    // Gibbs method. Calculates probability under each view
    void __transitionColumnAssignmentGibbs(size_t which_column, size_t m=1);
    void __transitionColumnAssignmentGibbsBootstrap(size_t which_column,
                                                    size_t m=1);
    void __transitionColumnAssignmentEnumeration(size_t which_column);

    // probability and sample helpers
    double __doPredictiveLogpObserved(size_t row, size_t column, double value);
    double __doPredictiveLogpUnobserved(size_t column, double value);
    double __doPredictiveDrawObserved(size_t row, size_t col);
    double __doPredictiveDrawUnobserved(size_t col); // FIXME: implement!

    // Cleanup methods
    void __destroySingletonView(size_t feature_index, size_t to_destroy,
                                size_t move_to);
    void __swapSingletonViews(size_t feature_index, size_t view_index, 
                              View &proposal_view);
    void __createSingletonView(size_t feature_index, size_t current_view_index,
                               View &proposal_view);
    void __moveFeatureToView(size_t feature_index, size_t move_from,
                             size_t move_to );

    void __insertConstraints(std::vector<std::vector<size_t>> indices,
                             std::vector<double> values);
    void __removeConstraints(std::vector<std::vector<size_t>> indices, 
                             std::vector<double> values);

    // MEMEBERS
    // data table size
    size_t _num_rows;
    size_t _num_columns;

    // CRP parameter
    double _crp_alpha;
    // for geweke. Will be set to baxcat::geweke_default_alpha if doing geweke
    // with no column hypers transitions (each view's CRP alpha will default to
    // 1), or will be -1(draw from prior) otherwise.
    double _view_alpha_marker;
    // and its config {shape, scale}
    std::vector<double> _crp_alpha_config;

    // partition information
    // number of views
    size_t _num_views;
    // Nv[v] is the number of features assigned to View v
    std::vector<size_t> _view_counts;
    // Zv[f] is the feature to which Feature f is assigned
    std::vector<size_t> _column_assignment;

    // Parallel random number generator
    std::shared_ptr<baxcat::PRNG> _rng;

    // holds the views
    std::vector<View> _views;

    // hold pointers to the features. They're shared_ptr's because features are
    // also owned by Views.
    std::vector<std::shared_ptr<BaseFeature>> _features;

    std::vector<datatype> _feature_types;

};


} // end namespace baxcat
#endif


// BaxCat: an extensible cross-catigorization engine.
// Copyright (C) 2014 Baxter Eaves
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License (LICENSE.txt) along with this
// program. If not, see <http://www.gnu.org/licenses/>.
//
// You may contact the mantainers of this software via github
// <https://github.com/BaxterEaves/baxcat_cxx>.

#include "state.hpp"

using std::vector;
using std::string;
using std::shared_ptr;


namespace baxcat{


// todo: add more complete constructors (alphas)
State::State(vector<vector<double>> X, vector<string> datatypes, vector<vector<double>> distargs,
             unsigned int rng_seed )
    : _rng(shared_ptr<PRNG>(new PRNG(rng_seed))), _crp_alpha_config({1, 1}), _view_alpha_marker(-1)
{
    _num_columns = X.size();
    _num_rows = X[0].size();
    _feature_types = helpers::getDatatypes(datatypes);
    _features = helpers::genFeatures(X, datatypes, distargs, _rng.get());

    // generate alpha
    _crp_alpha = _rng->gamrand(_crp_alpha_config[0], _crp_alpha_config[1]);

    // generate partitions;
    _rng.get()->crpGen(_crp_alpha, _num_columns, _column_assignment, _num_views, _view_counts);

    // create views
    vector<vector<shared_ptr<BaseFeature>>> view_features(_num_views);

    for( size_t i = 0; i < _num_columns; ++i){
        auto v = _column_assignment[i];
        view_features[v].push_back(_features[i]);
    }

    for(size_t v = 0; v < _num_views; ++v)
        _views.push_back( View(view_features[v], _rng.get()) );
}


State::State(vector<vector<double>> X, vector<string> datatypes, vector<vector<double>> distargs,
             unsigned int rng_seed, vector<size_t> Zv, vector<vector<size_t>> Zrcv,
             vector<map<string, double>> hypers_maps)
    : _column_assignment(Zv), _rng(shared_ptr<PRNG>(new PRNG(rng_seed))), 
      _crp_alpha_config({1, 1}), _view_alpha_marker(-1)
{
    _num_columns = X.size();
    _num_rows = X[0].size();

    _feature_types = helpers::getDatatypes(datatypes);
    _features = helpers::genFeatures(X, datatypes, distargs, _rng.get());

    _crp_alpha = _rng->gamrand(_crp_alpha_config[0], _crp_alpha_config[1]);

    _num_views = utils::vector_max(_column_assignment)+1;
    _view_counts.resize(_num_views,0);
    for(auto z : _column_assignment)
        ++_view_counts[z];

    // create views
    vector<vector<shared_ptr<BaseFeature>>> view_features(_num_views);
    for(size_t i = 0; i < _num_columns; ++i){
        auto v = _column_assignment[i];
        view_features[v].push_back(_features[i]);
    }

    for(size_t v = 0; v < _num_views; ++v)
        _views.push_back(View(view_features[v], _rng.get(), -1, Zrcv[v]));

    if(!hypers_maps.empty()){
        assert(hypers_maps.size() == _features.size());
        for (size_t f=0; f < hypers_maps.size(); ++f) {
            _features[f].get()->setHypers(hypers_maps[f]);
        }
    }
}


State::State(size_t num_rows, vector<string> datatypes, vector<vector<double>> distargs, 
             bool fix_hypers, bool fix_row_alpha, bool fix_col_alpha)
    : _num_rows(num_rows), _num_columns(datatypes.size()),  _rng(shared_ptr<PRNG>(new PRNG()))
{
    _crp_alpha_config = {5, .2};

    _view_alpha_marker = fix_row_alpha ? baxcat::geweke_default_alpha : -1;

    if(fix_col_alpha){
        _crp_alpha = baxcat::geweke_default_alpha;
    }else{
        _crp_alpha = _rng->gamrand(_crp_alpha_config[0], _crp_alpha_config[1]);
    }

    // generate partitions
    _rng.get()->crpGen(_crp_alpha, _num_columns, _column_assignment, _num_views, _view_counts);

    vector<vector<double>> X;
    vector<double> Y(num_rows, NAN);
    for(size_t col = 0; col < _num_columns; ++col)
        X.push_back(Y);

    _feature_types = helpers::getDatatypes(datatypes);
    _features = helpers::genFeatures(X, datatypes, distargs, _rng.get(), true, fix_hypers);

    // create views
    vector<vector<shared_ptr<BaseFeature>>> view_features(_num_views);

    for( size_t i = 0; i < _num_columns; ++i){
        auto v = _column_assignment[i];
        view_features[v].push_back(_features[i]);
    }

    for(size_t v = 0; v < _num_views; ++v)
        _views.push_back(View(view_features[v], _rng.get(), _view_alpha_marker, {}));
}

// probability
//`````````````````````````````````````````````````````````````````````````````````````````````````
// TODO: dependent queries
vector<double> State::predictiveLogp(vector<vector<size_t>> query_indices,
                                     vector<double> query_values,
                                     vector<vector<size_t>> constraint_indices,
                                     vector<double> constraint_values)
{
    size_t n_queries = query_indices.size();

    // insert all the constraint values
    if(not constraint_indices.empty())
        __insertConstraints(constraint_indices, constraint_values);

    // do probabilities
    vector<double> logps(n_queries, 0);
    for(size_t q = 0; q < n_queries; ++q){
        auto row = query_indices[q][0];
        auto col = query_indices[q][1];
        auto val = query_values[q];
        double logp;
        if(row < _num_rows){
            logp = __doPredictiveLogpObserved(row, col, val);
        }else{
            logp = __doPredictiveLogpUnobserved(col, val);
        }
        logps[q] = logp;
    }

    // clean up
    if(not constraint_indices.empty())
        __removeConstraints(constraint_indices, constraint_values);

    return logps;
}


double State::__doPredictiveLogpObserved(size_t row, size_t col, double val)
{
    auto type = _feature_types[col];
    auto view = _column_assignment[col];
    auto cluster_idx = _views[view].getAssignmentOfRow(row);
    if(helpers::is_discrete(type)){
        size_t casted_value = size_t(val);
        return _features[col].get()->valueLogp(casted_value, cluster_idx);
    }else{
        return _features[col].get()->valueLogp(val, cluster_idx);
    }
}


double State::__doPredictiveLogpUnobserved(size_t col, double val)
{
    auto type = _feature_types[col];
    auto view = _column_assignment[col];
    auto crp_alpha = _views[view].getCRPAlpha();
    auto num_categories = _views[view].getNumCategories();
    auto category_counts = _views[view].getClusterCounts();

    vector<double> logps(num_categories+1, 0);

    double crp_denom = double(_num_rows+1) + crp_alpha;
    double log_weight;
    // iterate through clusters
    for(size_t k = 0; k < num_categories; ++k){
        log_weight = log(double(category_counts[k])/crp_denom);
        if(helpers::is_discrete(type)){
            size_t casted_value = size_t(val);
            logps[k] = _features[col].get()->valueLogp(casted_value, k)+log_weight;
        }else{
            logps[k] = _features[col].get()->valueLogp(val, k)+log_weight;
        }
    }

    log_weight = log(crp_alpha/crp_denom);

    if(helpers::is_discrete(type)){
        // FIXME: variable casting
        size_t casted_value = size_t(val);
        logps[num_categories] = _features[col].get()->singletonValueLogp(casted_value)+log_weight;
    }else{
        logps[num_categories] = _features[col].get()->singletonValueLogp(val)+log_weight;
    }

    return numerics::logsumexp(logps);
}


// sample (NOT REFACTORED)
//`````````````````````````````````````````````````````````````````````````````````````````````````
vector<vector<double>> State::predictiveDraw(vector<vector<size_t>> query_indices,
                                             vector<vector<size_t>> constraint_indices,
                                             vector<double> constraint_values, size_t N)
{
    size_t n_queries = query_indices.size();

    // allocate the output vector
    vector<vector<double>> samples(N, vector<double>(n_queries, 0));

    // insert all the constraint values
    if(constraint_indices.size() > 0)
        __insertConstraints(constraint_indices, constraint_values);

    for(size_t q = 0; q < n_queries; ++q){
        auto row = query_indices[q][0];
        auto col = query_indices[q][1];
        bool is_observed = (row < _num_rows);
        for(size_t n = 0; n < N; ++n){
            double sample;
            if(is_observed){
                sample = __doPredictiveDrawObserved(row, col);
            }else{
                // we have to do this relative to the other queries
                sample = __doPredictiveDrawUnobserved(col);
            }
            samples[n][q] = sample;
        }
    }

    // clean up
    if(constraint_indices.size() > 0)
        __removeConstraints(constraint_indices, constraint_values);

    return samples;
}


// FIXME: implement
double  State::__doPredictiveDrawObserved(size_t row, size_t col)
{
    auto view = _column_assignment[col];
    auto cluster_idx = _views[view].getAssignmentOfRow(row);
    double draw = _features[col].get()->drawFromCluster(cluster_idx, _rng.get());
    return draw;
}


// FIXME: implement
double  State::__doPredictiveDrawUnobserved(size_t col)
{
    return -1.*double(col+.5);
}


// Transition helpers
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::transition(vector< string > which_transitions, vector<size_t> which_rows,
                       vector<size_t> which_cols, size_t which_kernel, int N)
{
    // conver strings to transitions
    vector<transition_type> t_list;
    bool do_shuffle = false;
    if(which_transitions.empty()){
        do_shuffle = true;
        t_list = helpers::all_transitions;
    }else{
        t_list = helpers::getTransitions(which_transitions);
    }

    for(int i = 0; i < N; ++i){
        if (do_shuffle)
            t_list = _rng.get()->shuffle(t_list);

        for( auto transition: t_list)
            __doTransition(transition, which_rows, which_cols, which_kernel);
    }
}


void State::__doTransition(transition_type t, vector<size_t> which_rows, vector<size_t> which_cols,
                           size_t which_kernel)
{
    switch(t){
        case transition_type::row_assignment:
            // std::cout << "Doing row_z" << std::endl;
            __transitionRowAssignments(which_rows);
            break;
        case transition_type::column_assignment:
            // std::cout << "Doing col_z" << std::endl;
            __transitionColumnAssignment(which_cols, which_kernel);
            break;
        case transition_type::row_alpha:
            // std::cout << "Doing row_alphas" << std::endl;
            __transitionViewCRPAlphas();
            break;
        case transition_type::column_alpha:
            // std::cout << "Doing col_alphas" << std::endl;
            __transitionStateCRPAlpha();
            break;
        case transition_type::column_hypers:
            // std::cout << "Doing col_z" << std::endl;
            __transitionColumnHypers(which_cols);
            break;
    }
}


// Transitions
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::__transitionStateCRPAlpha()
{
    // don't worry about CRP alpha is there is only one column
    if(_num_columns == 1) return;

    // construct crp alpha posterior
    double k = _num_views;
    double n = _num_columns;

    double alpha_shape = _crp_alpha_config[0];
    double alpha_scale = _crp_alpha_config[1];

    auto log_crp_posterior = [alpha_shape, alpha_scale, k, n](double x){
        return numerics::lcrpUNormPost(k, n, x) + dist::gamma::logPdf(x, alpha_shape, alpha_scale);
    };

    double slice_width = alpha_shape*alpha_scale*alpha_scale/2;  // this is a guess
    size_t burn = 25;

    // slice sample
    _crp_alpha = samplers::sliceSample(_crp_alpha, log_crp_posterior, {0, INF},
                                       slice_width, burn, _rng.get());
}


void State::__transitionViewCRPAlphas()
{
    // when to do in parallel?
    for(auto &view: _views)
        view.transitionCRPAlpha();
}


void State::__transitionColumnHypers(vector<size_t> which_cols)
{
    if(which_cols.empty()){
        // #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < _num_columns; i++)
            _features[i].get()->updateHypers();
    }else{
        // #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < which_cols.size(); ++i){
            auto col = which_cols[i];
            _features[col].get()->updateHypers();
        }
    }
}


void State::__transitionRowAssignments(vector<size_t> which_rows)
{
    // #pragma omp parallel for schedule(static)
    for(size_t v = 0; v < _num_views; ++v){
        if( which_rows.empty() ){
            _views[v].transitionRows();
        }else{
            for(auto r : which_rows)
                _views[v].transitionRow(r);
        }
    }
}


void State::__transitionColumnAssignment(vector<size_t> which_cols, size_t which_kernel)
{

    // don't transition columns if there is only one
    if(_num_columns == 1) return;

    if(which_cols.empty()){
        which_cols.resize(_num_columns, 0);
        for(size_t i = 0; i < _num_columns; ++i)
            which_cols[i] = i;
        which_cols = _rng.get()->shuffle(which_cols);
    }

    if(which_kernel == 0){
        for(auto col : which_cols)
            __transitionColumnAssignmentGibbs(col);
    }else{
        // FIXME: proper exceptino
        throw 1;
    }
}


// column transition kernels
// ````````````````````````````````````````````````````````````````````````````````````````````````
void State::__transitionColumnAssignmentGibbs(size_t col, size_t m)
{
    // double log_crp_denom = log(double(_num_columns-1) + _crp_alpha);

    auto view_index_current = _column_assignment[col];
    bool is_singleton = (_view_counts[view_index_current] == 1);

    vector<double> log_crps(_num_views,0);
    for(size_t v = 0; v < _num_views; ++v)
        log_crps[v] = log(double(_view_counts[v]));

    if (is_singleton){
        log_crps[view_index_current] = log(_crp_alpha);
    }else{
        --log_crps[view_index_current];
        log_crps.push_back(_crp_alpha);
    }

    // TODO: optimization: preallocate
    vector<double> logps;

    auto feature = _features[col];

    for(size_t v = 0; v < _num_views; ++v){
        feature.get()->reassign(_views[v].getRowAssignments());
        double logp = feature.get()->logp()+log_crps[v];
        logps.push_back(logp);
    }

    // if this is not already a singleton view, we must propose a singleton
    if(!is_singleton){
        vector<shared_ptr<BaseFeature>> fvec = {feature};
        vector<View> view_holder;
        double log_m = log(static_cast<double>(m));
        for(size_t i = 0; i < m; ++i){
            View proposal_view(fvec, _rng.get(), _view_alpha_marker, {});
            view_holder.push_back(proposal_view);
            double logp = feature.get()->logp()+log_crps.back()-log_m;
            logps.push_back(logp);
        }

        auto view_index_new = _rng.get()->lpflip(logps);

        if (view_index_new != view_index_current){
            if (view_index_new >= _num_views){
                auto proposal_view = view_holder[view_index_new-_num_views];
                __createSingletonView(col, view_index_current, proposal_view);
            }else{
                __moveFeatureToView(col, view_index_current, view_index_new);
            }
        }else{
            feature.get()->reassign(_views[view_index_current].getRowAssignments());
        }
    }else{
        auto view_index_new = _rng.get()->lpflip(logps);
        if (view_index_new != view_index_current){
            __destroySingletonView(col, view_index_current, view_index_new);
        }else{
            feature.get()->reassign(_views[view_index_current].getRowAssignments());
        }
    }
}


// Cleanup
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::__destroySingletonView(size_t feat_idx, size_t to_destroy, size_t move_to)
{
    _column_assignment[feat_idx] = move_to;
    _views[to_destroy].releaseFeature(feat_idx);
    for(size_t i = 0; i < _num_columns; ++i)
        _column_assignment[i] -= (_column_assignment[i] > to_destroy) ? 1 : 0;

    _views[move_to].assimilateFeature(_features[feat_idx]);
    ++_view_counts[move_to];
    _view_counts.erase(_view_counts.begin()+to_destroy);
    _views.erase(_views.begin()+to_destroy);
    --_num_views;
}


void State::__swapSingletonViews(size_t feat_idx, size_t view_index, View proposal_view)
{
    _views[view_index] = proposal_view;
    _features[feat_idx].get()->reassign(proposal_view.getRowAssignments());
}


void State::__createSingletonView(size_t feat_idx, size_t current_view_index, View proposal_view)
{
    _column_assignment[feat_idx] = _num_views;
    _features[feat_idx].get()->reassign(proposal_view.getRowAssignments());
    _views[current_view_index].releaseFeature(feat_idx);
    --_view_counts[current_view_index];
    _view_counts.push_back(1);
    _views.push_back(proposal_view);
    ++_num_views;
}


void State::__moveFeatureToView(size_t feat_idx, size_t move_from, size_t move_to)
{
    _column_assignment[feat_idx] = move_to;
    _views[move_from].releaseFeature(feat_idx);
    --_view_counts[move_from];
    _views[move_to].assimilateFeature(_features[feat_idx]);
    ++_view_counts[move_to];
}


// add/remove constraint_values
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::__insertConstraints(vector<vector<size_t>> indices, vector<double> values)
{
    size_t n_constraints = values.size();
    for(size_t i = 0; i < n_constraints; ++i){
        auto row = indices[i][0];
        auto col = indices[i][1];
        auto view = _column_assignment[col];
        auto cluster_idx = _views[view].getAssignmentOfRow(row);
        auto type = _feature_types[col];
        // do cast and insert
        if(helpers::is_discrete(type)){
            // FIXME: add casting to different types
            size_t casted_value = size_t(values[i]);
            _features[col].get()->insertValue(casted_value, cluster_idx);
        }else{
            _features[col].get()->insertValue(values[i], cluster_idx);
        }
    }
}


void State::__removeConstraints(vector<vector<size_t>> indices, vector<double> values)
{
    size_t n_constraints = values.size();
    for(size_t i = 0; i < n_constraints; ++i){
        auto row = indices[i][0];
        auto col = indices[i][1];
        auto view = _column_assignment[col];
        auto cluster_idx = _views[view].getAssignmentOfRow(row);
        auto type = _feature_types[col];
        // do cast and remove
        if(helpers::is_discrete(type)){
            // FIXME: add casting to different types
            size_t casted_value = size_t(values[i]);
            _features[col].get()->removeValue(casted_value, cluster_idx);
        }else{
            _features[col].get()->removeValue(values[i], cluster_idx);
        }
    }
}


//append
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::appendRow(std::vector<double> data_row, bool assign_to_max_p_cluster)
{
    for(auto &view : _views){
        auto feature_indices = view.getFeatureIndices();
        vector<double> data;
        vector<size_t> indices;
        for( auto i : feature_indices){
            data.push_back(data_row[i]);
            indices.push_back(i);
        }
        view.appendRow(data, indices, assign_to_max_p_cluster);
    }
}


void State::popRow()
{
    for(auto &view : _views)
        view.popRow();
}


// getters
//`````````````````````````````````````````````````````````````````````````````````````````````````
vector<size_t> State::getColumnAssignment() const
{
    return _column_assignment;
}


vector<vector<size_t>> State::getRowAssignments() const
{
    vector<vector<size_t>> row_assignment;
    for(auto &view : _views)
        row_assignment.push_back(view.getRowAssignments());

    return row_assignment;
}


vector<map<string, double>> State::getColumnHypers() const
{
    vector<map<string, double>> column_hypers;
    for(auto &feature : _features)
        column_hypers.push_back(feature.get()->getHypersMap());

    return column_hypers;
}


vector<vector<map<string, double>>> State::getSuffstats() const
{
    vector<vector<map<string, double>>> feature_suffstats;
    for(auto &feature : _features)
        feature_suffstats.push_back(feature.get()->getModelSuffstats());

    return feature_suffstats;
}

vector<double> State::getViewCRPAlphas() const
{
    vector<double> view_alphas;
    for(auto &view : _views)
        view_alphas.push_back(view.getCRPAlpha());

    return view_alphas;
}


double State::getStateCRPAlpha() const
{
    return _crp_alpha;
}


// setters
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::setHyperConfig(size_t column_index, std::vector<double> hyperprior_config)
{
    _features[column_index].get()->setHyperConfig(hyperprior_config);
}


void State::setHypers(size_t column_index, std::map<std::string, double> hypers_map)
{
    _features[column_index].get()->setHypers(hypers_map);
}


void State::setHypers(size_t column_index, std::vector<double> hypers_vec)
{
    _features[column_index].get()->setHypers(hypers_vec);
}


// geweke
//`````````````````````````````````````````````````````````````````````````````````````````````````
void State::__geweke_clear()
{
    for(size_t f = 0; f < _num_columns; ++f)
        _features[f].get()->__geweke_clear();
}


vector<double> State::__geweke_pullDataColumn(size_t column_index) const
{
    // TODO: Figure out how to return variable types for different feature types rather than casting
    // in Container
    auto data_temp = _features[column_index].get()->getData();
    return data_temp;
}


void State::__geweke_resampleRow(size_t which_row)
{
    for(size_t f = 0; f < _num_columns; ++f){
        size_t view_index = _column_assignment[f];
        size_t category_index = _views[view_index].getAssignmentOfRow(which_row);
        _features[f].get()->__geweke_resampleRow(which_row, category_index, _rng.get());
    }
}


void State::__geweke_resampleRows()
{

    vector<size_t> rows(_num_rows);
    for(size_t i = 0; i < _num_rows; i++)
        rows[i] = i;

    rows = _rng->shuffle(rows);

    for(size_t i = 0; i < _num_rows; i++)
        __geweke_resampleRow(rows[i]);
}


} // end namespace baxcat

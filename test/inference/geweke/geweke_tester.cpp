
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

#include "geweke_tester.hpp"

using std::vector;
using std::string;
using std::map;

namespace baxcat {

GewekeTester::GewekeTester(size_t num_rows, size_t num_cols, vector<string> datatypes, 
                           unsigned int seed, size_t m, bool do_hypers, 
                           bool do_row_alpha, bool do_col_alpha,
                           bool do_row_z, bool do_col_z, size_t ct_kernel)
    : _m(m), _num_cols(num_cols), _num_rows(num_rows), _datatypes(datatypes),
      _do_hypers(do_hypers), _ct_kernel(ct_kernel),
      _do_row_alpha(do_row_alpha), _do_col_alpha(do_col_alpha), 
      _do_col_z(do_col_z), _do_row_z(do_row_z)
{

    ASSERT(std::cout, _num_cols >= 1);
    ASSERT(std::cout, _num_rows > 1);
    
    _seeder.seed(seed);

    printf("Constructing args...");
    for(size_t i = 0; i < _num_cols; ++i)
        _distargs.push_back(baxcat::geweke_default_distargs[datatypes[i]]);

    printf("done.\n");

    printf("Intializing State...");
    _state =  State(_num_rows, _datatypes, _distargs, !do_hypers, !do_row_alpha, !do_col_alpha,
                    !do_row_z, !do_col_z);
    printf("done.\n");

    _transition_list = {};

    if(_do_row_z) _transition_list.push_back("row_assignment");
    if(_do_col_z) _transition_list.push_back("column_assignment");
    if(_do_hypers) _transition_list.push_back("column_hypers");
    if(do_row_alpha) _transition_list.push_back("row_alpha");
    if(do_col_alpha) _transition_list.push_back("column_alpha");

    printf("Intialized.\n");
}


void GewekeTester::forwardSample(size_t num_times, bool do_init)
{
    std::uniform_int_distribution<unsigned int> urnd;
    if(do_init)
        __initStats(_state, _state_crp_alpha_forward, _all_stats_forward, _num_views_forward);

    for(size_t i = 0; i < num_times; ++i){
        if( !((i+1) % 5))
            printf("\rSample %zu of %zu", i+1, num_times); fflush(stdout);

        _state = State(_num_rows, _datatypes, _distargs, !_do_hypers, 
                       !_do_row_alpha, !_do_col_alpha, !_do_row_z, !_do_col_z);

        // _state.__geweke_initHypers();
    
        _state.__geweke_clear();
        _state.__geweke_resampleRows();
        __updateStats(_state, _state_crp_alpha_forward, _all_stats_forward, _num_views_forward);
    }
    printf("\n");
}


void GewekeTester::posteriorSample(size_t num_times, bool do_init, size_t lag)
{
    if(do_init)
        __initStats(_state, _state_crp_alpha_posterior, _all_stats_posterior, 
                    _num_views_posterior);

    // forward sample
    _state = State(_num_rows, _datatypes, _distargs, !_do_hypers, !_do_row_alpha, !_do_col_alpha,
                   !_do_row_z, !_do_col_z);

    // _state.__geweke_initHypers();

    _state.__geweke_clear();
    _state.__geweke_resampleRows();
    // do a bunch of posterior samples
    for(size_t i = 0; i < num_times; ++i){
        if( !((i+1) % 5))
            printf("\rSample %zu of %zu", i+1, num_times); fflush(stdout);

        for( size_t j = 0; j < lag; ++j ){
            _state.transition(_transition_list, {}, {}, _ct_kernel, 1, _m);
            _state.__geweke_resampleRows();
        }

        __updateStats(_state, _state_crp_alpha_posterior, _all_stats_posterior, 
                      _num_views_posterior);
    }
    printf("\n");
}


// Helpers
//`````````````````````````````````````````````````````````````````````````````````````````````````
template <typename T>
vector<string> GewekeTester::__getMapKeys(map<string, T> map_in)
{
    vector<string> keys;
    for(auto imap : map_in)
        keys.push_back(imap.first);

    return keys;
}


template <typename T>
vector<double> GewekeTester::__getDataStats(const vector<T> &data, size_t categorical_K)
{
    vector<double> stats;
    if(categorical_K > 0){
        stats.push_back(test_utils::chi2Stat(data, categorical_K));
    }else{
        double mean = utils::vector_mean(data);
        double var = 0;
        for(auto &x : data)
            var += (mean-x)*(mean-x);
        double norm = static_cast<double>(data.size());
        stats.push_back(mean);
        stats.push_back(sqrt(var/norm));
    }
    return stats;
}


void GewekeTester::__updateStats(const State &state, vector<double> &state_crp_alpha,
                                 vector<map<string, vector<double>>> &all_stats,
                                 vector<size_t> &num_views)
{
    num_views.push_back(state.getNumViews());
    state_crp_alpha.push_back(state.getStateCRPAlpha());
    auto column_hypers = state.getColumnHypers();

    for(size_t i = 0; i < column_hypers.size(); ++i)
    {
        auto hyper_keys = GewekeTester::__getMapKeys(column_hypers[i]);
        string categorial_marker = "dirichlet_alpha";
        bool is_categorial = test_utils::hasElement(hyper_keys, categorial_marker) == 1;
        size_t categorical_k = is_categorial ? 5 : 0;
        auto data = state.__geweke_pullDataColumn(i);

        ASSERT_EQUAL(std::cout, data.size(), _num_rows);

        auto data_stat = GewekeTester::__getDataStats(data, categorical_k);

        if(is_categorial){
            all_stats[i]["chi-square"].push_back(data_stat[0]);
        }else{
            all_stats[i]["mean"].push_back(data_stat[0]);
            all_stats[i]["std"].push_back(data_stat[1]);
        }

        if(_do_hypers)
            for(auto &hyper_key : hyper_keys)
                all_stats[i][hyper_key].push_back(column_hypers[i][hyper_key]);
    }
}


void GewekeTester::__initStats(const State &state, vector<double> &state_crp_alpha,
                               vector<map<string, vector<double>>> &all_stats,
                               vector<size_t> &num_views)
{
    state_crp_alpha = {};
    num_views = {};
    auto column_hypers = state.getColumnHypers();

    all_stats.resize(column_hypers.size());

    for(size_t i = 0; i < column_hypers.size(); ++i)
    {
        auto hyper_keys = GewekeTester::__getMapKeys(column_hypers[i]);
        string categorial_marker = "dirichlet_alpha";
        bool is_categorial = test_utils::hasElement(hyper_keys,categorial_marker) == 1;

        if(is_categorial){
            all_stats[i]["chi-square"] = {};
        }else{
            all_stats[i]["mean"] = {};
            all_stats[i]["std"] = {};
        }

        if(_do_hypers)
            for(auto &hyper_key : hyper_keys)
                all_stats[i][hyper_key] = {};
    }
}


void GewekeTester::run(size_t num_times, size_t num_posterior_chains, size_t lag=25)
{
    assert(lag >= 1);

    size_t samples_per_chain = num_times/num_posterior_chains;
    std::cout << "Running forward samples" << std::endl;
    forwardSample(num_times, true);

    std::cout << "Running posterior samples (1 of " << num_posterior_chains << ")"  << std::endl;
    posteriorSample(samples_per_chain, true, lag);
    for( size_t i = 0; i < num_posterior_chains-1; ++i){
        std::cout << "Running posterior samples (" << i+2 << " of ";
        std::cout << num_posterior_chains << ")"  << std::endl;
        posteriorSample(samples_per_chain, false, lag);
    }

    std::cout << "done." << std::endl;
}


// Test results output and plotting
//`````````````````````````````````````````````````````````````````````````````````````````````````
void GewekeTester::outputResults()
{
    size_t num_pass = 0;
    size_t num_fail = 0;
    bool all_pass = true;

    for( size_t i = 0; i < _num_cols; ++i){
        // get stat keys
        std::cout << "COLUMN " << i << std::endl;
        auto keys = GewekeTester::__getMapKeys(_all_stats_forward[i]);
        mglGraph gr;
        int plots_y = 3;
        int plots_x = 0;

        for(auto key : keys)
            plots_x += (_all_stats_forward[i][key].size() == 0) ? 0 : 1;

        std::stringstream filename;
        filename << "results/column_" << i << ".png";
        gr.SetSize(500*plots_x,500*plots_y);

        int index = 0;
        for(auto key : keys){
            // std::cout << "outputting " << key << std::endl;
            int pp_plot_index = index;
            int forward_hist_index = index + plots_x;
            int posterior_hist_index = index + 2*plots_x;

            std::stringstream test_name;
            test_name << "column " << i << " " << key;

            std::stringstream ss;
            ss << "ks-test column " << i << " [" << key << "]";

            // std::cout << "\tgetting data: " << key << std::endl;
            size_t n_forward = _all_stats_forward[i][key].size();
            size_t n_posterior = _all_stats_posterior[i][key].size();

            // std::cout << "\tks-test: " << key << std::endl;
            gr.SubPlot(plots_x, plots_y, pp_plot_index);
            auto ks_stat = test_utils::twoSampleKSTest(_all_stats_forward[i][key],
                _all_stats_posterior[i][key], true, &gr, test_name.str());

            // std::cout << "\tgresults: " << key << std::endl;
            bool distributions_differ = test_utils::ksTestRejectNull(ks_stat, n_forward, n_posterior);
            test_utils::__output_ks_test_result(distributions_differ, ks_stat, ss.str());
            test_utils::__update_pass_counters(num_pass, num_fail, all_pass, !distributions_differ);

            gr.SubPlot(plots_x, plots_y, forward_hist_index);
            baxcat::plotting::hist(&gr, _all_stats_forward[i][key], 31, "forward");

            gr.SubPlot(plots_x, plots_y, posterior_hist_index);
            baxcat::plotting::hist(&gr, _all_stats_posterior[i][key], 31, "posterior");

            index++;

        }
        gr.WriteFrame(filename.str().c_str());
    }

    size_t plots_x = 0;
    if(_do_col_z and _do_col_alpha and _num_cols > 1){
        plots_x = 2;
    }else if(_do_col_z){
        plots_x = 1;
    }

    mglGraph gr;

    gr.SetSize(500*plots_x,500*2);

    size_t n_forward = 0;
    size_t n_posterior = 0;
    size_t plot_num = 0;

    if(_do_col_z and _num_cols > 1){
        printf("plot num views.\n");
        n_forward = _num_views_forward.size();
        n_posterior = _num_views_posterior.size();
        // auto chi2Stat = test_utils::chi2gof(_num_views_posterior, _num_views_forward);
        // TODO: p-value and output
        
        gr.SubPlot(plots_x, 2, 0);
        baxcat::plotting::hist(&gr, _num_views_forward, "V forward", _num_cols+1);

        gr.SubPlot(plots_x, 2, 1);
        baxcat::plotting::hist(&gr, _num_views_posterior, "V posterior", _num_cols+1);

        plot_num = 2;
    }

    if(_do_col_alpha and _do_col_z and _num_cols > 1){
        printf("plot col alpha.\n");
        std::stringstream ss;
        ss << "ks-test [state alpha]";
        n_forward = _state_crp_alpha_forward.size();
        n_posterior = _state_crp_alpha_posterior.size();
        auto ks_stat = test_utils::twoSampleKSTest(_state_crp_alpha_forward, _state_crp_alpha_posterior);
        bool distributions_differ = test_utils::ksTestRejectNull(ks_stat, n_forward, n_posterior);
        test_utils::__output_ks_test_result(distributions_differ, ks_stat, ss.str());
        test_utils::__update_pass_counters(num_pass, num_fail, all_pass, !distributions_differ);

        gr.SubPlot(plots_x, 2, plot_num);
        baxcat::plotting::hist(&gr, _state_crp_alpha_forward, 31, "State alpha forward");

        gr.SubPlot(plots_x, 2, plot_num+1);
        baxcat::plotting::hist(&gr, _state_crp_alpha_posterior, 31, "State alpha posterior");
    }

    if(plots_x > 0){
        printf("plot state.\n");
        gr.WriteFrame("results/state.png");
    }

    if(all_pass){
        std::cout << "**No failures detected." << std::endl;
    }else{
        std::cout << "**" << num_fail << " failures." << std::endl;
    }
}

} // end namespace

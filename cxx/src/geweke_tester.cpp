
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

    if(_do_col_alpha) _transition_list.push_back("column_alpha");
    if(_do_col_z) _transition_list.push_back("column_assignment");
    if(_do_row_alpha) _transition_list.push_back("row_alpha");
    if(_do_row_z) _transition_list.push_back("row_assignment");
    if(_do_hypers) _transition_list.push_back("column_hypers");

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
    _state = State(_num_rows, _datatypes, _distargs, !_do_hypers,
                   !_do_row_alpha, !_do_col_alpha, !_do_row_z, !_do_col_z);

    // _state.__geweke_initHypers();

    _state.__geweke_clear();
    _state.__geweke_resampleRows();
    // do a bunch of posterior samples
    for(size_t i = 0; i < num_times; ++i){
        if( !((i+1) % 5))
            printf("\rSample %zu of %zu", i+1, num_times); fflush(stdout);

        for( size_t j = 0; j < lag; ++j ){
            _state.transition(_transition_list, vector<size_t>(), vector<size_t>(),
                              _ct_kernel, 1, _m);
            _state.__geweke_clear();
            _state.__geweke_resampleRows();
        }

        __updateStats(_state, _state_crp_alpha_posterior, _all_stats_posterior,
                      _num_views_posterior);
    }

    printf("\n");
}


// Helpers
//`````````````````````````````````````````````````````````````````````````````
template <typename T>
vector<string> GewekeTester::__getMapKeys(map<string, T> map_in)
{
    vector<string> keys;
    for(auto imap : map_in)
        keys.push_back(imap.first);

    return keys;
}


template <typename T>
vector<double> GewekeTester::__getDataStats(const vector<T> &data,
        size_t categorical_K)
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


void GewekeTester::__updateStats(const State &state,
        vector<double> &state_crp_alpha,
        vector<map<string, vector<double>>> &all_stats,
        vector<size_t> &num_views)
{
    // if we're not doing col_z, we should take stats on row_z (there should
    // be only one view)
    if(_do_col_z){
        num_views.push_back(state.getNumViews());
        state_crp_alpha.push_back(state.getStateCRPAlpha());    
    }else{
        ASSERT_EQUAL(std::cout, state.getNumViews(), 1);

        auto row_assignment = state.getRowAssignments()[0];
        auto num_categories = *std::max_element(row_assignment.begin(),
                                                row_assignment.end())+1;
        auto view_alphas = state.getViewCRPAlphas();

        ASSERT_EQUAL(std::cout, view_alphas.size(), 1);

        num_views.push_back(num_categories);
        state_crp_alpha.push_back(view_alphas[0]);
    }
    
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


void GewekeTester::__initStats(const State &state,
        vector<double> &state_crp_alpha, vector<map<string,
        vector<double>>> &all_stats, vector<size_t> &num_views)
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


void GewekeTester::run(size_t num_times, size_t num_posterior_chains,
        size_t lag=25)
{
    ASSERT(std::cout, lag >= 1);

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

    // do some checks
    if(not _do_col_z) ASSERT_EQUAL(std::cout, _state.getNumViews(), 1);

    if(not _do_row_z) ASSERT_EQUAL(std::cout, _state.getNumViews(), 1);

    if(not _do_col_alpha){
        ASSERT_EQUAL(std::cout, _state.getStateCRPAlpha(),
                     baxcat::geweke_default_alpha);
    }else{
        ASSERT_NOT_EQUAL(std::cout, _state.getStateCRPAlpha(),
                         baxcat::geweke_default_alpha);
    }

    for(auto &alpha : _state.getViewCRPAlphas()){
        if(not _do_row_alpha){
            ASSERT_EQUAL(std::cout, alpha, baxcat::geweke_default_alpha);
        }else{
            ASSERT_NOT_EQUAL(std::cout, alpha, baxcat::geweke_default_alpha);
        }
    }
}


vector<map<string, vector<double>>> GewekeTester::getForwardStats()
{
    return _all_stats_forward;
}


vector<map<string, vector<double>>> GewekeTester::getPosteriorStats()
{
    return _all_stats_posterior;
}



vector<double> GewekeTester::getStateAlphaForward()
{
    return _state_crp_alpha_forward;
}


vector<double> GewekeTester::getStateAlphaPosterior()
{
    return _state_crp_alpha_posterior;
}


vector<size_t> GewekeTester::getNumViewsForward()
{
    return _num_views_forward;
}


vector<size_t> GewekeTester::getNumViewsPosterior()
{
    return _num_views_posterior;
}

} // end namespace

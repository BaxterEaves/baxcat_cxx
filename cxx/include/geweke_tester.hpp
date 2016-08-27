
#ifndef baxcat_cxx_geweke_tester_guard
#define baxcat_cxx_geweke_tester_guard

#include <map>
#include <string>
#include <vector>
#include <random>
#include <cassert>
#include <utility>
#include <mgl2/mgl.h> // MathGL plotting

#include "state.hpp"
#include "utils.hpp"
#include "plotting.hpp"
#include "test_utils.hpp"
#include "helpers/synthetic_data_generator.hpp"
#include "template_helpers.hpp"

namespace baxcat{

class GewekeTester
{
public:
    // generate state with all continuous data
    GewekeTester(size_t num_rows, size_t num_cols, std::vector<std::string> datatypes, 
                 unsigned int seed, size_t m=1, bool do_hypers=true, bool do_row_alpha=true,
                 bool do_col_alpha=true, bool do_row_z=true, bool do_col_z=true, 
                 size_t ct_kernel=0);

    void run(size_t num_times, size_t num_posterior_chains, size_t lag);

    void forwardSample(size_t num_times, bool do_init);
    void posteriorSample(size_t num_times, bool do_init, size_t lag);
    int outputResults();

    template <typename T>
    std::vector<std::string> __getMapKeys(std::map<std::string, T> map_in);

    template <typename T>
    std::vector<double> __getDataStats(const std::vector<T> &data, size_t categorial_k);

    void __updateStats( const baxcat::State &state, std::vector<double> &state_crp_alpha,
        std::vector<std::map<std::string, std::vector<double>>> &all_stats,
        std::vector<size_t> &num_views);

    void __initStats( const baxcat::State &state, std::vector<double> &state_crp_alpha,
        std::vector<std::map<std::string, std::vector<double>>> &all_stats,
        std::vector<size_t> &num_views);

private:
    std::vector<std::string> _transition_list;
    
    size_t _num_cols;
    size_t _num_rows;

    size_t _m;
    size_t _ct_kernel;

    bool _do_hypers;
    bool _do_row_alpha;
    bool _do_col_alpha;
    bool _do_row_z;
    bool _do_col_z;

    baxcat::State _state;

    std::vector<std::vector<double>> _distargs;
    std::vector<std::string> _datatypes;

    std::vector<double> _state_crp_alpha_forward;
    std::vector<double> _state_crp_alpha_posterior;
    std::vector<size_t> _num_views_forward;
    std::vector<size_t> _num_views_posterior;
    std::vector<std::map<std::string, std::vector<double>>> _all_stats_forward;
    std::vector<std::map<std::string, std::vector<double>>> _all_stats_posterior;

    std::mt19937 _seeder;
};

} // end namespace

#endif

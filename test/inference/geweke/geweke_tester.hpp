#ifndef baxcat_cxx_geweke_tester_guard
#define baxcat_cxx_geweke_tester_guard

#include <string>
#include <vector>
#include <random>
#include <cassert>
#include <utility>
#include <mgl2/mgl.h> // MathGL plotting
#include <map>

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
    GewekeTester(size_t num_rows, size_t num_cols, unsigned int seed);

    // TODO: implement generate state with mixed datatypes
    // GewekeTester(size_t num_rows, std::vector<string> datatypes, size_t seed);

    void run(size_t num_times, size_t num_posterior_chains, size_t lag);

    void forwardSample(size_t num_times, bool do_init);
    void posteriorSample(size_t num_times, bool do_init, size_t lag);
    void outputResults();

    template <typename T>
    static std::vector<std::string> __getMapKeys(std::map<std::string, T> map_in);

    template <typename T>
    static std::vector<double> __getDataStats(const std::vector<T> &data, bool is_categorial);

    static void __updateStats( const baxcat::State &state, std::vector<double> &state_crp_alpha,
        std::vector<std::map<std::string, std::vector<double>>> &all_stats);

    static void __initStats( const baxcat::State &state, std::vector<double> &state_crp_alpha,
        std::vector<std::map<std::string, std::vector<double>>> &all_stats);
    
private:
    size_t _num_cols;

    baxcat::State _state;

    std::vector<std::vector<double>> _seed_data;
    std::vector<std::vector<double>> _seed_args;
    std::vector<std::string> _datatypes;

    std::vector<double> _state_crp_alpha_forward;
    std::vector<double> _state_crp_alpha_posterior;
    std::vector<std::map<std::string, std::vector<double>>> _all_stats_forward;
    std::vector<std::map<std::string, std::vector<double>>> _all_stats_posterior;
    
    std::mt19937 _seeder;
};

} // end namespace

#endif
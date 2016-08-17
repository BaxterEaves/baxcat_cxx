
#ifndef baxcat_cxx_constants_guard
#define baxcat_cxx_constants_guard

#include <map>
#include <string>
#include <vector>

namespace baxcat{

// Indicate which transition to do
enum transition_type{
    row_assignment,
    column_assignment,
    row_alpha,
    column_alpha,
    column_hypers
};

// Which model to init
enum datatype{
    continuous,     // normal, normal-gamma
    categorical,    // multinomial, dirichlet
    binomial,       // bernoulli, beta
    count,          // poisson, gamma
    magnitude,      // lognormal, normal-gamma
    bounded,        // beta, gamma-exponential
    cyclic,         // von mises, von misses-inverse gamma
};

static std::map<std::string, std::vector<double>> geweke_default_hypers = {
    {"continuous", {0, 1, 10, 10}},
    {"categorical", {1}}
};

static std::map<std::string, std::vector<double>> geweke_default_hyperprior_config = {
    {"continuous", {0, .1, 10, .1, 40, .25, 40, .25}},
    {"categorical", {10}}
};

static std::map<std::string, std::vector<double>> geweke_default_distargs = {
    {"continuous", {0}},
    {"categorical", {5}}
};

static double geweke_default_alpha = 1;

}

#endif

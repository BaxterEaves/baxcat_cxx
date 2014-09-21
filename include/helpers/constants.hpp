#ifndef baxcat_cxx_constants_guard
#define baxcat_cxx_constants_guard

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

}

#endif
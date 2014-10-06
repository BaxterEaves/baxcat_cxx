
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

#include "datatypes/continuous.hpp"

using std::map;
using std::cout;
using std::vector;
using std::string;
using std::function;
using baxcat::samplers::sliceSample;
using baxcat::datatypes::Continuous;


void Continuous::insertElement(double x)
{
    ASSERT_IS_A_NUMBER(cout, x);

    ++_n;
    _nng.suffstatInsert(x, _sum_x, _sum_x_sq);

    ASSERT_IS_A_NUMBER(cout, _sum_x);
    ASSERT_IS_A_NUMBER(cout, _sum_x_sq);
    ASSERT_INFO(cout, "Invalid suffstat", !(_n==1 && (_sum_x != 0 && _sum_x_sq == 0)) );

    // optimization note: Strategically move updateConstants
    updateConstants();
}

void Continuous::removeElement(double x)
{
    ASSERT_IS_A_NUMBER(cout, x);

    --_n;
    // protect from floating point errors where _sum_x_sq == 0
    if( _n == 0 ){
        _sum_x = 0;
        _sum_x_sq = 0;
    }else if (_n==1){
        _sum_x -= x;
        _sum_x_sq = _sum_x*_sum_x;
    }else{
        _nng.suffstatRemove( x, _sum_x, _sum_x_sq );
    }

    ASSERT_IS_A_NUMBER(cout, _sum_x);
    ASSERT_IS_A_NUMBER(cout, _sum_x_sq);
    ASSERT_INFO(cout, "Invalid suffstat", !(_n==1 && (_sum_x != 0 && _sum_x_sq == 0)));

    // optimization note: Strategically move updateConstants
    updateConstants();
}


void Continuous::clear(const std::vector<double> &distargs)
{
    _n = 0;
    _sum_x = 0;
    _sum_x_sq = 0;
}


double Continuous::logp() const
{
    return _nng.logMarginalLikelihood(_n, _log_ZN, _log_Z0);
}


double Continuous::elementLogp(double x) const
{
    return _nng.logPredictiveProbability(x, _n, _sum_x, _sum_x_sq, _m, _r,  _s, _nu, _log_ZN);
}


double Continuous::singletonLogp(double x) const
{
    return _nng.logPredictiveProbability(x, 0, 0, 0, _m, _r, _s, _nu, _log_Z0);
}


double Continuous::draw(baxcat::PRNG *rng) const
{
    double sample = _nng.predictiveSample(_n, _sum_x, _sum_x_sq, _m, _r, _s, _nu, rng);
    ASSERT_IS_A_NUMBER(cout, sample);
    return sample;
}


double Continuous::drawConstrained( vector<double> contraints, baxcat::PRNG *rng ) const
{
    double n = _n;
    double sum_x = _sum_x;
    double sum_x_sq = _sum_x_sq;

    n += static_cast<double>(contraints.size());

    for( auto &x : contraints )
        _nng.suffstatInsert(x, sum_x, sum_x_sq);

    double sample = _nng.predictiveSample( n, sum_x, sum_x_sq, _m, _r, _s, _nu, rng);
    ASSERT_IS_A_NUMBER(cout, sample);
    return sample;
}


// hyperprior management
// ````````````````````````````````````````````````````````````````````````````````````````````````
vector<double> Continuous::initHypers( const vector<double> &hyperprior_config, baxcat::PRNG *rng )
{
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[M_STD]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SCALE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SCALE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[NU_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[NU_SCALE]);

    double m_mean = hyperprior_config[M_MEAN];
    double m_std = hyperprior_config[M_STD];
    double m = rng->normrand(m_mean, m_std);

    double r_shape = hyperprior_config[R_SHAPE];
    double r_scale = hyperprior_config[R_SCALE];
    double r = rng->gamrand(r_shape, r_scale);

    double s_shape = hyperprior_config[S_SHAPE];
    double s_scale = hyperprior_config[S_SCALE];
    double s = rng->gamrand(s_shape, s_scale);

    double nu_shape = hyperprior_config[NU_SHAPE];
    double nu_scale = hyperprior_config[NU_SCALE];
    double nu = rng->gamrand(nu_shape, nu_scale);

    vector<double> hypers(4);
    hypers[HYPER_M] = m;
    hypers[HYPER_R] = r;
    hypers[HYPER_S] = s;
    hypers[HYPER_NU] = nu;

    return hypers;
}


vector<double> Continuous::constructHyperpriorConfig(const vector<double> &X)
{
    double mean_x = baxcat::utils::vector_mean(X);
    double std_x = sqrt( baxcat::utils::sum_of_squares(X)/static_cast<double>(X.size()) );

    ASSERT_GREATER_THAN_ZERO(cout, std_x);

    vector<double> config(8,1);
    config[M_MEAN] = mean_x;
    config[M_STD] = std_x;
    config[R_SHAPE] = 1;
    config[R_SCALE] = std_x;
    config[S_SHAPE] = 1;
    config[S_SCALE] = std_x;
    config[NU_SHAPE] = 2;
    config[NU_SCALE] = .5;

    return config;
}


vector<double> Continuous::resampleHypers(vector<Continuous> &models,
                                          const vector<double> &hyperprior_config, 
                                          baxcat::PRNG *rng, size_t burn)
{
    ASSERT_EQUAL(std::cout, hyperprior_config.size(), 8);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[M_STD]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SCALE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SCALE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[NU_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[NU_SCALE]);

    // construct sampler equations
    auto m_unscaled_posterior = constructMConditional(models, hyperprior_config);
    auto r_unscaled_posterior = constructRConditional(models, hyperprior_config);
    auto s_unscaled_posterior = constructSConditional(models, hyperprior_config);
    auto nu_unscaled_posterior = constructNuConditional(models, hyperprior_config);

    // get initial hypers
    auto hypers = models[0].getHypers();

    // starting point and expected slice width
    double x_0, w;

    double U = rng->urand(-1, 1);

    // resample m
    w = hyperprior_config[M_STD]/2.0;
    x_0 = hyperprior_config[M_MEAN] + U*w;
    hypers[HYPER_M] = sliceSample(x_0, m_unscaled_posterior, {-INF, INF}, w, burn, rng);

    // Note: setHypers updates log_Z0 and log_ZN
    for(auto &model : models)
        model.setHypers(hypers);

    w = hyperprior_config[R_SHAPE]*hyperprior_config[R_SCALE]*hyperprior_config[R_SCALE]/2;
    U = rng->urand(-1,1);
    x_0 = fabs(hyperprior_config[R_SCALE] + U*w);
    hypers[HYPER_R] = sliceSample(x_0, r_unscaled_posterior, {ALMOST_ZERO, INF}, w, burn, rng);

    for(auto &model : models)
        model.setHypers(hypers);

    w = hyperprior_config[S_SHAPE]*hyperprior_config[S_SCALE]*hyperprior_config[S_SCALE]/2;
    U = rng->urand(-1,1);
    x_0 = fabs(hyperprior_config[S_SCALE] + U*w);
    hypers[HYPER_S] = sliceSample(x_0, s_unscaled_posterior, {ALMOST_ZERO, INF}, w, burn, rng);

    for(auto &model : models)
        model.setHypers(hypers);

    w = hyperprior_config[NU_SHAPE]*hyperprior_config[NU_SCALE]*hyperprior_config[NU_SCALE]/2;
    U = rng->urand(-1,1);
    x_0 = fabs(hypers[HYPER_NU] + U*w);
    hypers[HYPER_NU] = sliceSample(x_0, nu_unscaled_posterior, {ALMOST_ZERO, INF}, w, burn, rng);

    for(auto &model : models)
        model.setHypers(hypers);

    return hypers;
}


// single-cluster hyperparameter conditionals
// ````````````````````````````````````````````````````````````````````````````````````````````````
double Continuous::hyperMConditional_(double m) const
{
    // we can cache log_Z0 in this case because m is not a factor
    return _nng.logMarginalLikelihood( _n, _sum_x, _sum_x_sq, m, _r, _s, _nu, _log_Z0 );
}


double Continuous::hyperRConditional_(double r) const
{
    // ASSERT_GREATER_THAN_ZERO(cout, r);
    return _nng.logMarginalLikelihood( _n, _sum_x, _sum_x_sq, _m, r, _s, _nu );
}


double Continuous::hyperSConditional_(double s) const
{
    // ASSERT_GREATER_THAN_ZERO(cout, s);
    return _nng.logMarginalLikelihood( _n, _sum_x, _sum_x_sq, _m, _r, s,  _nu );
}


double Continuous::hyperNuConditional_(double nu) const
{
    // ASSERT_GREATER_THAN_ZERO(cout, nu);
    return _nng.logMarginalLikelihood( _n, _sum_x,  _sum_x_sq, _m, _r, _s, nu );
}


double Continuous::hyperpriorLogp(const std::vector<double> &hyperprior_config) const
{
    ASSERT_EQUAL(std::cout, hyperprior_config.size(), 8);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[M_STD]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SCALE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SCALE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[NU_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[NU_SCALE]);

    double logp = 0;
    double m_rho = 1.0/(hyperprior_config[M_STD]*hyperprior_config[M_STD]);
    logp += baxcat::dist::gaussian::logPdf(_m, hyperprior_config[M_MEAN], m_rho);
    logp += baxcat::dist::gamma::logPdf(_r, hyperprior_config[R_SHAPE], hyperprior_config[R_SCALE]);
    logp += baxcat::dist::gamma::logPdf(_s, hyperprior_config[S_SHAPE], hyperprior_config[S_SCALE]);
    logp += baxcat::dist::gamma::logPdf(_nu, hyperprior_config[NU_SHAPE], hyperprior_config[NU_SCALE]);
    return logp;
}


// Construct hyperparameter conditionals (unscaled)
// ````````````````````````````````````````````````````````````````````````````````````````````````
function<double(double)> Continuous::constructMConditional(const vector<Continuous> &models, 
                                                           const vector<double> &hyperprior_config)
{
    ASSERT_EQUAL(std::cout, hyperprior_config.size(), 8);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[M_STD]);

    double m_mean = hyperprior_config[M_MEAN];
    double m_std = hyperprior_config[M_STD];
    auto m_unscaled_posterior = [m_mean, m_std, models](double m){
        double logp = baxcat::dist::gaussian::logPdf(m, m_mean, 1.0/(m_std*m_std));
        for(auto &model : models){
            logp += model.hyperMConditional_(m);
        }
        return logp;
    };
    return m_unscaled_posterior;
}


function<double(double)> Continuous::constructRConditional(const vector<Continuous> &models, 
                                                           const vector<double> &hyperprior_config)
{
    ASSERT_EQUAL(std::cout, hyperprior_config.size(), 8);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[R_SCALE]);

    double r_scale = hyperprior_config[R_SCALE];
    double r_shape = hyperprior_config[R_SHAPE];
    auto r_unscaled_posterior = [r_shape, r_scale, models](double r){
        double logp = baxcat::dist::gamma::logPdf(r, r_shape, r_scale);
        for(auto &model : models){
            logp += model.hyperRConditional_(r);
        }
        return logp;
    };
    return r_unscaled_posterior;
}


function<double(double)> Continuous::constructSConditional(const vector<Continuous> &models, 
                                                           const vector<double> &hyperprior_config)
{
    ASSERT_EQUAL(std::cout, hyperprior_config.size(), 8);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SCALE]);

    double s_shape = hyperprior_config[S_SHAPE];
    double s_scale = hyperprior_config[S_SCALE];
    auto s_unscaled_posterior = [s_shape, s_scale, models](double s){
        double logp = baxcat::dist::gamma::logPdf(s, s_shape, s_scale);
        for(auto &model : models){
            logp += model.hyperSConditional_(s);
        }
        return logp;
    };
    return s_unscaled_posterior;
}


function<double(double)> Continuous::constructNuConditional(const vector<Continuous> &models,
                                                            const vector<double> &hyperprior_config)
{
    ASSERT_EQUAL(std::cout, hyperprior_config.size(), 8);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SHAPE]);
    ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[S_SCALE]);

    double nu_shape = hyperprior_config[NU_SHAPE];
    double nu_scale = hyperprior_config[NU_SCALE];
    auto nu_unscaled_posterior = [nu_shape, nu_scale, models](double nu){
        double logp = baxcat::dist::gamma::logPdf(nu, nu_shape, nu_scale);
        for(auto &model : models){
            logp += model.hyperNuConditional_(nu);
        }
        return logp;
    };
    return nu_unscaled_posterior;
}


// updaters
// ````````````````````````````````````````````````````````````````````````````````````````````````
void Continuous::updateConstants()
{
    double m_n = _m;
    double r_n = _r;
    double s_n = _s;
    double nu_n = _nu;

    _nng.posteriorParameters(_n, _sum_x, _sum_x_sq, m_n, r_n, s_n, nu_n);

    _log_Z0 = _nng.logZ(_r, _s, _nu);
    _log_ZN = _nng.logZ(r_n, s_n, nu_n);

    ASSERT_IS_A_NUMBER(cout, _log_Z0);
    ASSERT_IS_A_NUMBER(cout, _log_ZN);
}


// setters and getters
// ````````````````````````````````````````````````````````````````````````````````````````````````
map<string, double> Continuous::getSuffstatsMap() const
{
    map<string, double> suffstats;
    suffstats["n"] = _n;
    suffstats["sum_x"] = _sum_x;
    suffstats["sum_x_sq"] = _sum_x_sq;
    return suffstats;
}


vector<double> Continuous::getHypers() const
{
    vector<double> hypers(4);
    hypers[HYPER_M] = _m;
    hypers[HYPER_R] = _r;
    hypers[HYPER_S] = _s;
    hypers[HYPER_NU] = _nu;
    return hypers;
}


map<string, double> Continuous::getHypersMap() const
{
    map<string, double> hypers;
    hypers["m"] = _m;
    hypers["r"] = _r;
    hypers["s"] = _s;
    hypers["nu"] = _nu;
    return hypers;
}


void Continuous::setHypers( vector<double> hypers )
{
    ASSERT_GREATER_THAN_ZERO( cout, hypers[HYPER_R] );
    ASSERT_GREATER_THAN_ZERO( cout, hypers[HYPER_S] );
    ASSERT_GREATER_THAN_ZERO( cout, hypers[HYPER_NU] );

    _m = hypers[HYPER_M];
    _r = hypers[HYPER_R];
    _s = hypers[HYPER_S];
    _nu = hypers[HYPER_NU];
    updateConstants(); // update normalizing constants
}


void Continuous::setHypersByMap( map<string, double> hypers )
{
    ASSERT_GREATER_THAN_ZERO( cout, hypers["r"] );
    ASSERT_GREATER_THAN_ZERO( cout, hypers["s"] );
    ASSERT_GREATER_THAN_ZERO( cout, hypers["nu"] );

    _m = hypers["m"];
    _r = hypers["r"];
    _s = hypers["s"];
    _nu = hypers["nu"];
    updateConstants(); // update normalizing constants
}

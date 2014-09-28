
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

#include "datatypes/categorical.hpp"

using std::map;
using std::cout;
using std::vector;
using std::string;
using std::function;
using baxcat::samplers::sliceSample;
using baxcat::datatypes::Categorical;

// cleanup
// ````````````````````````````````````````````````````````````````````````````````````````````````
void Categorical::insertElement(size_t x)
{
	++_n;
	_msd.suffstatInsert(x, _counts);
}


void Categorical::removeElement(size_t x)
{
	--_n;
	_msd.suffstatRemove(x, _counts);
}


void Categorical::clear(const std::vector<double> &distargs)
{
	_n = 0;
	std::fill (_counts.begin(), _counts.end(), 0);
}


// probability
// ````````````````````````````````````````````````````````````````````````````````````````````````
double Categorical::logp() const
{
    return _msd.logMarginalLikelihood(_n, _counts, _dirichlet_alpha);
}


double Categorical::elementLogp(size_t x) const
{
    return _msd.logPredictiveProbability(x, _counts, _dirichlet_alpha, _log_Z0 );
}


double Categorical::singletonLogp(size_t x) const
{
	return _msd.logSingletonProbability(x, _counts.size(), _dirichlet_alpha);
}


double Categorical::hyperpriorLogp(const std::vector<double> &hyperprior_config) const
{
	ASSERT_GREATER_THAN_ZERO(cout, hyperprior_config[DIRICHLET_ALPHA_SCALE]);

	return baxcat::dist::gamma::logPdf(_dirichlet_alpha, 1.,
									   hyperprior_config[DIRICHLET_ALPHA_SCALE]);
}

// draw
// ````````````````````````````````````````````````````````````````````````````````````````````````
size_t Categorical::draw(baxcat::PRNG *rng) const
{
	return _msd.predictiveSample(_counts, _dirichlet_alpha, rng, _log_Z0);
}


size_t Categorical::drawConstrained(vector<size_t> contraints, baxcat::PRNG *rng) const
{
	auto counts_copy = _counts;
	for( auto &c : contraints)
		++counts_copy[c];
	return _msd.predictiveSample(counts_copy, _dirichlet_alpha, rng, _log_Z0);
}


// hypers
// ````````````````````````````````````````````````````````````````````````````````````````````````
vector<double> Categorical::initHypers(const vector<double> &hyperprior_config, baxcat::PRNG *rng)
{
	double dirichlet_alpha = rng->gamrand(1., hyperprior_config[DIRICHLET_ALPHA_SCALE]);

	vector<double> hypers(1);
    hypers[HYPER_DIRICHLET_ALPHA] = dirichlet_alpha;

    return hypers;
}


vector<double> Categorical::constructHyperpriorConfig(const vector<double> &X)
{
	auto K = baxcat::utils::vector_max(X)+1;
	double alpha_scale = 1./K;

	vector<double> config(1, 0);
	config[DIRICHLET_ALPHA_SCALE] = alpha_scale;

	return config;
}


vector<double> Categorical::resampleHypers(vector<Categorical> &models,
    									   const vector<double> &hyperprior_config,
										   baxcat::PRNG *rng, size_t burn)
{
	// construct sampler equations
	auto alpha_unscaled_post = constructDirichletAlphaConditional(models, hyperprior_config);

	// get initial hypers
	auto hypers = models[0].getHypers();

	// starting point and expected slice width
    double x_0, w;

    double U = rng->urand(-1, 1);

    // resample m
    w = hyperprior_config[DIRICHLET_ALPHA_SCALE]/2.0;
    x_0 = hyperprior_config[DIRICHLET_ALPHA_SCALE] + U*w;
    hypers[HYPER_DIRICHLET_ALPHA] = sliceSample(x_0, alpha_unscaled_post, {0, INF}, w, burn, rng);

    // Note: setHypers updates log_Z0 and log_ZN
    for(auto &model : models)
        model.setHypers(hypers);

    return hypers;
}


// single-cluster hyperparameter conditionals
// ````````````````````````````````````````````````````````````````````````````````````````````````
double Categorical::hyperDirichletAlphaConditional_(double alpha) const
{
    return _msd.logMarginalLikelihood(_n, _counts, alpha);
}


// Construct hyperparameter conditionals (unscaled)
// ````````````````````````````````````````````````````````````````````````````````````````````````
function<double(double)> Categorical::constructDirichletAlphaConditional(
    const vector<Categorical> &models, const vector<double> &hyperprior_config)
{
    double alpha_scale = hyperprior_config[DIRICHLET_ALPHA_SCALE];
    auto alpha_unscaled_posterior = [alpha_scale, models](double alpha){
        double logp = baxcat::dist::gamma::logPdf(alpha, 1., alpha_scale);
        for( auto &model : models){
            logp += model.hyperDirichletAlphaConditional_(alpha);
        }
        return logp;
    };
    return alpha_unscaled_posterior;
}


// updaters
// ````````````````````````````````````````````````````````````````````````````````````````````````
void Categorical::updateConstants()
{
    _log_Z0 = 0;
}


// setters and getters
// ````````````````````````````````````````````````````````````````````````````````````````````````
map<string, double> Categorical::getSuffstatsMap() const
{
    map<string,double> suffstats = {
        {"k", double(_counts.size())},
        {"n", double(_n)}
    };

    for(size_t i = 0; i < _counts.size(); ++i){
        std::ostringstream key;
        key << i;
        suffstats[key.str()] = _counts[i];
    }

    return suffstats;
}


vector<double> Categorical::getHypers() const
{
    vector<double> hypers(1);
    hypers[HYPER_DIRICHLET_ALPHA] = _dirichlet_alpha;
    return hypers;
}


map<string, double> Categorical::getHypersMap() const
{
    map<string, double> hypers;
    hypers["dirichlet_alpha"] = _dirichlet_alpha;
    return hypers;
}


void Categorical::setHypers(vector<double> hypers)
{
    _dirichlet_alpha = hypers[HYPER_DIRICHLET_ALPHA];
    updateConstants(); // update normalizing constants
}


void Categorical::setHypersByMap( map<string, double> hypers )
{
    _dirichlet_alpha = hypers["dirichlet_alpha"];
    updateConstants(); // update normalizing constants
}

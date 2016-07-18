
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

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "prng.hpp"
#include "models/msd.hpp"
#include "distributions/gamma.hpp"

#include "datatypes/categorical.hpp"

using baxcat::datatypes::Categorical;

BOOST_AUTO_TEST_SUITE(categorical_test)

BOOST_AUTO_TEST_CASE(default_constructor_values_test)
{
    // creates a model with K=4
    std::vector<double> distargs = {4};
    Categorical model(distargs);

    auto suffstats = model.getSuffstatsMap();
    auto hypers = model.getHypersMap();

    // REQUIRE default suffstat values
    BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);

    // check default suffstat values
    BOOST_CHECK_EQUAL( hypers["dirichlet_alpha"], 1);
}


BOOST_AUTO_TEST_CASE(explicit_constructor_values_test)
{
    // creates a model with K=4
    Categorical model(10, {2, 3, 5}, 2.5);

    auto suffstats = model.getSuffstatsMap();
    auto hypers = model.getHypersMap();

    // REQUIRE default suffstat values
    BOOST_REQUIRE_EQUAL( suffstats["n"], 10);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 3);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 2);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 3);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 5);

    // check default suffstat values
    BOOST_CHECK_EQUAL( hypers["dirichlet_alpha"], 2.5);
}


// check  setters
BOOST_AUTO_TEST_CASE(set_hypers_should_set_values)
{
    std::vector<double> distargs = {4};
    Categorical model(distargs);


    model.setHypers({1.2});
    auto hypers = model.getHypers();
    BOOST_REQUIRE_EQUAL(hypers[0], 1.2);

    model.setHypers({.23});
    hypers = model.getHypers();
    BOOST_REQUIRE_EQUAL(hypers[0], .23);
}


BOOST_AUTO_TEST_CASE(set_hypers_by_map_should_set_values)
{
    std::vector<double> distargs = {4};
    Categorical model(distargs);

    std::map<std::string, double> hypers_map;
    hypers_map["dirichlet_alpha"] = 5;

    model.setHypersByMap(hypers_map);
    auto hypers = model.getHypers();
    BOOST_REQUIRE_EQUAL(hypers[0], 5);

    hypers_map["dirichlet_alpha"] = 0.12;

    model.setHypersByMap(hypers_map);
    hypers = model.getHypers();
    BOOST_REQUIRE_EQUAL(hypers[0], 0.12);
}


BOOST_AUTO_TEST_CASE(insert_should_increment_values)
{
    std::vector<double> distargs = {4};
    Categorical model(distargs);

    size_t x = 2;

    model.insertElement(x);

    auto suffstats = model.getSuffstatsMap();

    BOOST_REQUIRE_EQUAL( suffstats["n"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 4);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);

    size_t y = 0;

    model.insertElement(y);

    suffstats = model.getSuffstatsMap();

    BOOST_REQUIRE_EQUAL( suffstats["n"], 2);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 4);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);
}


BOOST_AUTO_TEST_CASE(remove_should_decrement_values)
{
    std::vector<double> distargs = {4};
    Categorical model(distargs);

    size_t x = 2;
    size_t y = 0;

    model.insertElement(x);
    model.insertElement(y);

    auto suffstats = model.getSuffstatsMap();

    BOOST_REQUIRE_EQUAL( suffstats["n"], 2);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 4);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);

    model.removeElement(y);
    model.removeElement(x);

    suffstats = model.getSuffstatsMap();

    BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 4);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);
}


BOOST_AUTO_TEST_CASE(clear_should_set_suffstats_to_zero)
{
    std::vector<double> distargs = {4};
    Categorical model(distargs);

    size_t x = 2;
    size_t y = 0;

    model.insertElement(x);
    model.insertElement(y);

    auto suffstats = model.getSuffstatsMap();

    BOOST_REQUIRE_EQUAL( suffstats["n"], 2);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 4);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 1);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);

    model.clear({4});

    suffstats = model.getSuffstatsMap();

    BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["k"], 4);
    BOOST_REQUIRE_EQUAL( suffstats["0"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["1"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["2"], 0);
    BOOST_REQUIRE_EQUAL( suffstats["3"], 0);
}


// Value checks
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(marginal_likelihood_value_checks)
{
    double n = 10;
    double dirichlet_alpha = 1.2;
    std::vector<size_t> counts = {2, 3, 5};
    Categorical model(n, counts, dirichlet_alpha);

    baxcat::models::MultinomialDirichlet<size_t> msd;

    double logp_model = model.logp();
    double logp_msd = msd.logMarginalLikelihood(n, counts, dirichlet_alpha);

    BOOST_REQUIRE_EQUAL(logp_msd, logp_model);
}


BOOST_AUTO_TEST_CASE(marginal_likelihood_value_checks_with_insert)
{

    double n = 10;
    double dirichlet_alpha = 1.2;
    std::vector<size_t> counts = {2, 3, 5};

    std::vector<size_t> X = {0, 0, 1, 1, 1, 2, 2, 2, 2, 2};
    Categorical model(0, {0, 0, 0}, dirichlet_alpha);

    for(auto x : X)
        model.insertElement(x);

    model.updateConstants();

    baxcat::models::MultinomialDirichlet<size_t> msd;

    double logp_model = model.logp();
    double logp_msd = msd.logMarginalLikelihood(n, counts, dirichlet_alpha);

    BOOST_REQUIRE_EQUAL(logp_msd, logp_model);
}


BOOST_AUTO_TEST_CASE(predictive_probability_value_checks)
{
    double n = 10;
    double dirichlet_alpha = 1.2;
    std::vector<size_t> counts = {2, 3, 5};
    Categorical model(n, counts, dirichlet_alpha);

    baxcat::models::MultinomialDirichlet<size_t> msd;
    double log_z = msd.logZ(n, counts, dirichlet_alpha);

    size_t x = 0;
    double logp_model = model.elementLogp(x);
    double logp_nng = msd.logPredictiveProbability(x, counts, dirichlet_alpha, log_z);
    BOOST_REQUIRE_EQUAL(logp_nng, logp_model);

    x = 1;
    logp_model = model.elementLogp(x);
    logp_nng = msd.logPredictiveProbability(x, counts, dirichlet_alpha, log_z);
    BOOST_REQUIRE_EQUAL(logp_nng, logp_model);
}


// single and multi model hyper parameter conditional values test
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(test_dirichlet_alpha_conditional_values_single)
{
    double n = 10;
    double dirichlet_alpha = 1.2;
    std::vector<size_t> counts = {2, 3, 5};

    std::vector<double> X = {0, 0, 1, 1, 1, 2, 2, 2, 2, 2};
    Categorical model(10, counts, dirichlet_alpha);

    auto config = Categorical::constructHyperpriorConfig(X);
    std::vector<Categorical> models = {model};
    auto a_conditional = Categorical::constructDirichletAlphaConditional(models, config);

    double a = 1.3;

    baxcat::models::MultinomialDirichlet<size_t> msd;

    auto hypers = model.getHypersMap();
    auto suffstats = model.getSuffstatsMap();

    double f_a = 0;
    f_a += baxcat::dist::gamma::logPdf(a, 1., config[0]);
    f_a += msd.logMarginalLikelihood(n, counts, a);

    BOOST_CHECK_EQUAL(a_conditional(a), f_a);
}


BOOST_AUTO_TEST_CASE(test_dirichlet_alpha_conditional_values_mulitple)
{
    double dirichlet_alpha = 1.2;

    double n_0 = 5;
    std::vector<size_t> counts_0 = {2, 3, 0};
    Categorical model_0(n_0, counts_0, dirichlet_alpha);

    double n_1 = 5;
    std::vector<size_t> counts_1 = {2, 3, 0};
    Categorical model_1(n_1, counts_1, dirichlet_alpha);

    std::vector<double> X = {0, 0, 1, 1, 1, 2, 2, 2, 2, 2};

    auto config = Categorical::constructHyperpriorConfig(X);
    std::vector<Categorical> models = {model_0, model_1};
    auto a_conditional = Categorical::constructDirichletAlphaConditional(models, config);

    double a = 1.3;

    baxcat::models::MultinomialDirichlet<size_t> msd;

    double f_a = 0;
    f_a += baxcat::dist::gamma::logPdf(a, 1., config[0]);
    f_a += msd.logMarginalLikelihood(n_0, counts_0, a);
    f_a += msd.logMarginalLikelihood(n_0, counts_0, a);

    BOOST_CHECK_EQUAL(a_conditional(a), f_a);
}


// Rsample hypers test
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(resample_hypers_should_change_hyper_values)
{
    double dirichlet_alpha = 1.2;
    std::vector<double> X = {0, 0, 1, 1, 1, 2, 2, 2, 2, 2};

    double n_0 = 5;
    std::vector<size_t> counts_0 = {2, 3, 0};
    Categorical model_0(n_0, counts_0, dirichlet_alpha);

    double n_1 = 5;
    std::vector<size_t> counts_1 = {2, 3, 0};
    Categorical model_1(n_1, counts_1, dirichlet_alpha);

    auto config = Categorical::constructHyperpriorConfig(X);
    std::vector<Categorical> models = {model_0, model_1};


    auto hypers = model_0.getHypersMap();

    static baxcat::PRNG *prng = new baxcat::PRNG();

    auto hypers_out = Categorical::resampleHypers(models, config, prng);

    BOOST_CHECK(hypers["dirichlet_alpha"] != hypers_out[0]);

    auto hypers_0 = models[0].getHypersMap();
    auto hypers_1 = models[1].getHypersMap();

    BOOST_CHECK_EQUAL(hypers_out[0], hypers_0["dirichlet_alpha"]);
    BOOST_CHECK_EQUAL(hypers_out[0], hypers_1["dirichlet_alpha"]);
}

BOOST_AUTO_TEST_SUITE_END()

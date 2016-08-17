
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "prng.hpp"
#include "models/nng.hpp"
#include "distributions/gamma.hpp"
#include "distributions/gaussian.hpp"
#include "distributions/inverse_gamma.hpp"

#include "datatypes/continuous.hpp"


BOOST_AUTO_TEST_SUITE(continuous_test)


BOOST_AUTO_TEST_CASE(default_constructor_values_test)
{
	baxcat::datatypes::Continuous model;

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypersMap();

	// REQUIRE default suffstat values
	BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], 0);

	// check default suffstat values
	BOOST_CHECK_EQUAL( hypers["m"], 0);
	BOOST_CHECK_EQUAL( hypers["r"], 1);
	BOOST_CHECK_EQUAL( hypers["s"], 1);
	BOOST_CHECK_EQUAL( hypers["nu"], 1);
}

BOOST_AUTO_TEST_CASE(distargs_constructor_values_test)
{
	std::vector<double> distargs;

	baxcat::datatypes::Continuous model(distargs);

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypersMap();

	// REQUIRE default suffstat values
	BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], 0);

	// check default suffstat values
	BOOST_CHECK_EQUAL( hypers["m"], 0);
	BOOST_CHECK_EQUAL( hypers["r"], 1);
	BOOST_CHECK_EQUAL( hypers["s"], 1);
	BOOST_CHECK_EQUAL( hypers["nu"], 1);
}

BOOST_AUTO_TEST_CASE(fill_in_constructor_values_test)
{
	double n = 4;
	double sum_x = 10;
	double sum_x_sq = 30;
	double m = 1;
	double r = 2;
	double s = 3;
	double nu = 4;

	baxcat::datatypes::Continuous model(n, sum_x, sum_x_sq, m, r, s, nu);

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypersMap();

	// REQUIRE default suffstat values
	BOOST_REQUIRE_EQUAL( suffstats["n"], n);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], sum_x);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], sum_x_sq);

	// check default suffstat values
	BOOST_REQUIRE_EQUAL( hypers["m"], m);
	BOOST_REQUIRE_EQUAL( hypers["r"], r);
	BOOST_REQUIRE_EQUAL( hypers["s"], s);
	BOOST_REQUIRE_EQUAL( hypers["nu"], nu);

	BOOST_REQUIRE_EQUAL( model.getCount(), 4);
}

BOOST_AUTO_TEST_CASE(copy_constructor_should_copy_values){
	double n = 4;
	double sum_x = 10;
	double sum_x_sq = 30;
	double m = 1;
	double r = 2;
	double s = 3;
	double nu = 4;

	baxcat::datatypes::Continuous model(n, sum_x, sum_x_sq, m, r, s, nu);
	baxcat::datatypes::Continuous model_copy(model);

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypers();

	auto suffstats_copy = model_copy.getSuffstatsMap();
	auto hypers_copy = model_copy.getHypers();

	BOOST_CHECK_EQUAL(suffstats["n"], suffstats_copy["n"]);
	BOOST_CHECK_EQUAL(suffstats["sum_x"], suffstats_copy["sum_x"]);
	BOOST_CHECK_EQUAL(suffstats["sum_x_sq"], suffstats_copy["sum_x_sq"]);

	BOOST_CHECK_EQUAL(hypers.size(), hypers_copy.size());
	BOOST_CHECK_EQUAL(hypers[0], hypers_copy[0]);
	BOOST_CHECK_EQUAL(hypers[1], hypers_copy[1]);
	BOOST_CHECK_EQUAL(hypers[2], hypers_copy[2]);
	BOOST_CHECK_EQUAL(hypers[3], hypers_copy[3]);
}

// check  setters
BOOST_AUTO_TEST_CASE(set_hypers_should_set_values)
{
	baxcat::datatypes::Continuous model;
	model.setHypers({5,6,7,8});

	auto hypers = model.getHypers();

	BOOST_REQUIRE_EQUAL(hypers[0], 5);
	BOOST_REQUIRE_EQUAL(hypers[1], 6);
	BOOST_REQUIRE_EQUAL(hypers[2], 7);
	BOOST_REQUIRE_EQUAL(hypers[3], 8);
}

BOOST_AUTO_TEST_CASE(set_hypers_by_map_should_set_values)
{
	baxcat::datatypes::Continuous model;

	std::map<std::string, double> hypers_map;
	hypers_map["m"] = 5;
	hypers_map["r"] = 6;
	hypers_map["s"] = 7;
	hypers_map["nu"] = 8;

	model.setHypersByMap(hypers_map);

	auto hypers = model.getHypers();

	BOOST_REQUIRE_EQUAL(hypers[0], 5);
	BOOST_REQUIRE_EQUAL(hypers[1], 6);
	BOOST_REQUIRE_EQUAL(hypers[2], 7);
	BOOST_REQUIRE_EQUAL(hypers[3], 8);
}

BOOST_AUTO_TEST_CASE(insert_should_increment_values)
{
	baxcat::datatypes::Continuous model;

	double x = 2;

	model.insertElement(x);

	auto suffstats = model.getSuffstatsMap();

	BOOST_REQUIRE_EQUAL( suffstats["n"], 1);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], x);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], x*x);

	double y = 4;

	model.insertElement(y);

	suffstats = model.getSuffstatsMap();

	BOOST_REQUIRE_EQUAL( suffstats["n"], 2);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], x+y);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], x*x + y*y);

}

BOOST_AUTO_TEST_CASE(remove_should_decrement_values)
{
	baxcat::datatypes::Continuous model;

	double x = 2;
	double y = 4;

	model.insertElement(x);
	model.insertElement(y);

	auto suffstats = model.getSuffstatsMap();

	BOOST_REQUIRE_EQUAL( suffstats["n"], 2);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], x+y);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], x*x + y*y);

	model.removeElement(y);
	model.removeElement(x);

	suffstats = model.getSuffstatsMap();

	BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], 0);

}

BOOST_AUTO_TEST_CASE(clear_should_set_suffstats_to_zero)
{
	baxcat::datatypes::Continuous model;

	double x = 2;
	double y = 4;

	model.insertElement(x);
	model.insertElement(y);

	auto suffstats = model.getSuffstatsMap();

	BOOST_REQUIRE_EQUAL( suffstats["n"], 2);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], x+y);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], x*x + y*y);

	model.clear(std::vector<double>());

	suffstats = model.getSuffstatsMap();

	BOOST_REQUIRE_EQUAL( suffstats["n"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x"], 0);
	BOOST_REQUIRE_EQUAL( suffstats["sum_x_sq"], 0);

}

// Value checks
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(marginal_likelihood_value_checks)
{
	baxcat::datatypes::Continuous model(4, 10, 30, 0, 1.2, 3, 2);

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypersMap();

	baxcat::models::NormalNormalGamma nng;

	double logp_model = model.logp();
	double logp_nng = nng.logMarginalLikelihood(suffstats["n"], suffstats["sum_x"],
		suffstats["sum_x_sq"], hypers["m"], hypers["r"], hypers["s"], hypers["nu"]);

	BOOST_REQUIRE_EQUAL(logp_nng, logp_model);

}

BOOST_AUTO_TEST_CASE(marginal_likelihood_value_checks_with_insert)
{
	std::vector<double> X = {1,2,3,4};
	baxcat::datatypes::Continuous model;

	for(auto x : X)
		model.insertElement(x);

	model.updateConstants();

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypersMap();

	baxcat::models::NormalNormalGamma nng;

	double logp_model = model.logp();
	double logp_nng = nng.logMarginalLikelihood(suffstats["n"], suffstats["sum_x"],
		suffstats["sum_x_sq"], hypers["m"], hypers["r"], hypers["s"], hypers["nu"]);

	BOOST_REQUIRE_EQUAL(logp_nng, logp_model);

}

BOOST_AUTO_TEST_CASE(predictive_probability_value_checks)
{
	baxcat::datatypes::Continuous model(4, 10, 30, 0, 1.2, 3, 2);

	auto suffstats = model.getSuffstatsMap();
	auto hypers = model.getHypersMap();

	baxcat::models::NormalNormalGamma nng;

	double x = 1.2;

	double logp_model = model.elementLogp(x);
	double logp_nng = nng.logPredictiveProbability(x, suffstats["n"], suffstats["sum_x"],
		suffstats["sum_x_sq"], hypers["m"], hypers["r"], hypers["s"], hypers["nu"]);

	BOOST_REQUIRE_EQUAL(logp_nng, logp_model);

}

// single model hyper parameter conditional values test
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(test_m_conditional_values_single)
{
	std::vector<double> X = {1,2,3,4};
	baxcat::datatypes::Continuous model(4, 10, 30, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model};
	auto m_conditional = baxcat::datatypes::Continuous::constructMConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model.getHypersMap();
	auto suffstats = model.getSuffstatsMap();

	double m_std = config[1];
	double m_mean = config[0];
	double f_m = 0;
	f_m += baxcat::dist::gaussian::logPdf(x, m_mean, 1/(m_std*m_std));
	f_m += nng.logMarginalLikelihood(suffstats["n"], suffstats["sum_x"], suffstats["sum_x_sq"], x,
									 hypers["r"], hypers["s"], hypers["nu"]);

	BOOST_CHECK_EQUAL(m_conditional(x), f_m);

}

BOOST_AUTO_TEST_CASE(test_r_conditional_values_single)
{
	std::vector<double> X = {1,2,3,4};
	baxcat::datatypes::Continuous model(4, 10, 30, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model};
	auto r_conditional = baxcat::datatypes::Continuous::constructRConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model.getHypersMap();
	auto suffstats = model.getSuffstatsMap();

	double f_r = 0;
	f_r += baxcat::dist::gamma::logPdf(x, config[2], config[3]);
	f_r += nng.logMarginalLikelihood(suffstats["n"], suffstats["sum_x"], suffstats["sum_x_sq"],
									 hypers["m"], x, hypers["s"], hypers["nu"]);

	BOOST_CHECK_EQUAL(r_conditional(x), f_r);
}

BOOST_AUTO_TEST_CASE(test_s_conditional_values_single)
{
	std::vector<double> X = {1,2,3,4};
	baxcat::datatypes::Continuous model(4, 10, 30, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model};
	auto s_conditional = baxcat::datatypes::Continuous::constructSConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model.getHypersMap();
	auto suffstats = model.getSuffstatsMap();

	double f_s = 0;
	f_s += baxcat::dist::gamma::logPdf(x, config[4], config[5]);
	f_s += nng.logMarginalLikelihood(suffstats["n"], suffstats["sum_x"], suffstats["sum_x_sq"],
									 hypers["m"], hypers["r"], x, hypers["nu"]);

	BOOST_CHECK_EQUAL(s_conditional(x), f_s);
}

BOOST_AUTO_TEST_CASE(test_nu_conditional_values_single)
{
	std::vector<double> X = {1,2,3,4};
	baxcat::datatypes::Continuous model(4, 10, 30, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model};
	auto nu_conditional = baxcat::datatypes::Continuous::constructNuConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model.getHypersMap();
	auto suffstats = model.getSuffstatsMap();

	double f_nu = 0;
	f_nu += baxcat::dist::gamma::logPdf(x, config[6], config[7]);
	f_nu += nng.logMarginalLikelihood(suffstats["n"], suffstats["sum_x"], suffstats["sum_x_sq"],
		hypers["m"], hypers["r"], hypers["s"], x);

	BOOST_CHECK_EQUAL(nu_conditional(x), f_nu);
}

// multiple model hyper parameter conditional values test
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(test_m_conditional_values_multiple)
{
	std::vector<double> X = {1,2,3,4,1,2,3,4,5};
	baxcat::datatypes::Continuous model_0(4, 10, 30, 0, 1.2, 3, 2);
	baxcat::datatypes::Continuous model_1(5, 15, 55, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models;// = {model_0, model_1};
	models.push_back(model_0);
	models.push_back(model_1);
	auto m_conditional = baxcat::datatypes::Continuous::constructMConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model_0.getHypersMap();
	auto suffstats_0 = model_0.getSuffstatsMap();
	auto suffstats_1 = model_1.getSuffstatsMap();

	double m_std = config[1];

	double f_m = 0;
	f_m += baxcat::dist::gaussian::logPdf(x, config[0], 1/(config[1]*config[1]));
	f_m += nng.logMarginalLikelihood(suffstats_0["n"], suffstats_0["sum_x"],
									 suffstats_0["sum_x_sq"], x, hypers["r"], hypers["s"],
									 hypers["nu"]);

	f_m += nng.logMarginalLikelihood(suffstats_1["n"], suffstats_1["sum_x"],
									 suffstats_1["sum_x_sq"], x, hypers["r"], hypers["s"],
									 hypers["nu"]);

	BOOST_CHECK_EQUAL(m_conditional(x), f_m);
}

BOOST_AUTO_TEST_CASE(test_r_conditional_values_multiple)
{
	std::vector<double> X = {1,2,3,4,1,2,3,4,5};
	baxcat::datatypes::Continuous model_0(4, 10, 30, 0, 1.2, 3, 2);
	baxcat::datatypes::Continuous model_1(5, 15, 55, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model_0, model_1};
	auto r_conditional = baxcat::datatypes::Continuous::constructRConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model_0.getHypersMap();
	auto suffstats_0 = model_0.getSuffstatsMap();
	auto suffstats_1 = model_1.getSuffstatsMap();

	double f_r = 0;
	f_r += baxcat::dist::gamma::logPdf(x, config[2], config[3]);
	f_r += nng.logMarginalLikelihood(suffstats_0["n"], suffstats_0["sum_x"],
		suffstats_0["sum_x_sq"], hypers["m"], x, hypers["s"], hypers["nu"]);
	f_r += nng.logMarginalLikelihood(suffstats_1["n"], suffstats_1["sum_x"],
		suffstats_1["sum_x_sq"], hypers["m"], x, hypers["s"], hypers["nu"]);

	BOOST_CHECK_EQUAL(r_conditional(x), f_r);
}

BOOST_AUTO_TEST_CASE(test_s_conditional_values_multiple)
{
	std::vector<double> X = {1,2,3,4,1,2,3,4,5};
	baxcat::datatypes::Continuous model_0(4, 10, 30, 0, 1.2, 3, 2);
	baxcat::datatypes::Continuous model_1(5, 15, 55, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model_0, model_1};
	auto s_conditional = baxcat::datatypes::Continuous::constructSConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model_0.getHypersMap();
	auto suffstats_0 = model_0.getSuffstatsMap();
	auto suffstats_1 = model_1.getSuffstatsMap();

	double f_s = 0;
	f_s += baxcat::dist::gamma::logPdf(x,config[4], config[5]);
	f_s += nng.logMarginalLikelihood(suffstats_0["n"], suffstats_0["sum_x"],
		suffstats_0["sum_x_sq"], hypers["m"], hypers["r"], x, hypers["nu"]);
	f_s += nng.logMarginalLikelihood(suffstats_1["n"], suffstats_1["sum_x"],
		suffstats_1["sum_x_sq"], hypers["m"], hypers["r"], x, hypers["nu"]);

	BOOST_CHECK_EQUAL(s_conditional(x), f_s);
}

BOOST_AUTO_TEST_CASE(test_nu_conditional_values_multiple)
{
	std::vector<double> X = {1,2,3,4,1,2,3,4,5};
	baxcat::datatypes::Continuous model_0(4, 10, 30, 0, 1.2, 3, 2);
	baxcat::datatypes::Continuous model_1(5, 15, 55, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model_0, model_1};
	auto nu_conditional = baxcat::datatypes::Continuous::constructNuConditional(models, config);

	double x = 1.3;

	baxcat::models::NormalNormalGamma nng;

	auto hypers = model_0.getHypersMap();
	auto suffstats_0 = model_0.getSuffstatsMap();
	auto suffstats_1 = model_1.getSuffstatsMap();

	double f_nu = 0;
	f_nu += baxcat::dist::gamma::logPdf(x, config[6], config[7]);
	f_nu += nng.logMarginalLikelihood(suffstats_0["n"], suffstats_0["sum_x"],
		suffstats_0["sum_x_sq"], hypers["m"], hypers["r"], hypers["s"], x);
	f_nu += nng.logMarginalLikelihood(suffstats_1["n"], suffstats_1["sum_x"],
		suffstats_1["sum_x_sq"], hypers["m"], hypers["r"], hypers["s"], x);

	BOOST_CHECK_EQUAL(nu_conditional(x), f_nu);
}


// sample test
BOOST_AUTO_TEST_CASE(predictive_sample_empty_suffstats_stress_test)
{
	baxcat::datatypes::Continuous model;
	static baxcat::PRNG *prng = new baxcat::PRNG();

	for(size_t i = 0; i < 100; ++i){
		double x = model.draw(prng);
		ASSERT_IS_A_NUMBER(std::cout, x);
	}
}

// Rsample hypers test
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(resample_hypers_should_change_hyper_values)
{
	std::vector<double> X = {1,2,3,4,1,2,3,4,5};
	baxcat::datatypes::Continuous model_0(4, 10, 30, 0, 1.2, 3, 2);
	baxcat::datatypes::Continuous model_1(5, 15, 55, 0, 1.2, 3, 2);
	auto config = baxcat::datatypes::Continuous::constructHyperpriorConfig(X);
	std::vector<baxcat::datatypes::Continuous> models = {model_0, model_1};

	auto hypers = model_0.getHypersMap();

	static baxcat::PRNG *prng = new baxcat::PRNG();

	auto hypers_out = baxcat::datatypes::Continuous::resampleHypers(models, config, prng);

	BOOST_CHECK(hypers["m"] != hypers_out[0]);
	BOOST_CHECK(hypers["r"] != hypers_out[1]);
	BOOST_CHECK(hypers["s"] != hypers_out[2]);
	BOOST_CHECK(hypers["nu"] != hypers_out[3]);

	auto hypers_0 = models[0].getHypersMap();

	BOOST_CHECK_EQUAL(hypers_out[0], hypers_0["m"]);
	BOOST_CHECK_EQUAL(hypers_out[1], hypers_0["r"]);
	BOOST_CHECK_EQUAL(hypers_out[2], hypers_0["s"]);
	BOOST_CHECK_EQUAL(hypers_out[3], hypers_0["nu"]);

	auto hypers_1 = models[1].getHypersMap();

	BOOST_CHECK_EQUAL(hypers_out[0], hypers_1["m"]);
	BOOST_CHECK_EQUAL(hypers_out[1], hypers_1["r"]);
	BOOST_CHECK_EQUAL(hypers_out[2], hypers_1["s"]);
	BOOST_CHECK_EQUAL(hypers_out[3], hypers_1["nu"]);

}

BOOST_AUTO_TEST_SUITE_END()

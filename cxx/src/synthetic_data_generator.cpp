
#include "helpers/synthetic_data_generator.hpp"

using std::vector;
using std::string;
using std::shared_ptr;
using std::map;


namespace baxcat{


SyntheticDataGenerator::SyntheticDataGenerator(size_t num_rows, vector<string> datatypes,
											   unsigned int rng_seed)
	: _rng(shared_ptr<PRNG>(new PRNG(rng_seed)))
{
	_num_columns = datatypes.size();
	_num_rows = num_rows;

	// get number of views
	size_t num_views = _rng.get()->randuint(_num_columns)+1;

	// get view weights
	_view_weights.resize(num_views, 1.0/static_cast<double>(num_views));

	// get number of categories in each view
	size_t max_categories = static_cast<size_t>(num_rows/5.0);
	vector<size_t> num_categories_in_views(num_views);
	for( auto &num_categories : num_categories_in_views )
		num_categories = _rng.get()->randuint(max_categories)+1;

	// generate cluster weights
	for(size_t i = 0; i < num_views; ++i){
		double uniform_weight = 1.0/static_cast<double>(num_categories_in_views[i]);
		vector<double> weights(num_categories_in_views[i], uniform_weight);
		_category_weights.push_back(weights);
	}

	// generate category separations
	for(size_t i = 0; i < _num_columns; ++i)
		_category_separation.push_back(_rng.get()->rand());

	_datatypes = helpers::getDatatypes(datatypes);

	this->initialize();
}


SyntheticDataGenerator::SyntheticDataGenerator(size_t num_rows, vector<double> view_weights,
											   vector<vector<double>> category_weights,
											   vector<double> category_separation,
											   vector<string> datatypes, unsigned int rng_seed)
	: _rng(shared_ptr<PRNG>(new PRNG(rng_seed)))
{
	_num_columns = category_separation.size();
	_num_rows = num_rows;
	_category_weights = category_weights;
	_view_weights = view_weights;
	_category_separation = category_separation;
	_datatypes = helpers::getDatatypes(datatypes);

	// TODO: input validation

	this->initialize();
}


void SyntheticDataGenerator::initialize()
{
	// generate partition of columns into views and rows in views into categories
	_column_assignment = generateWeightedParition(_view_weights, _num_rows);
	for(auto &w : _category_weights)
		_row_assignments.push_back(generateWeightedParition(w, _num_rows));

	double num_categories, separation;
	size_t this_view;

	for(size_t col = 0; col < _num_columns; ++col){
		auto type = _datatypes[col];
		this_view = _column_assignment[col];
		num_categories = _category_weights[this_view].size();
		separation = _category_separation[col];

		if(type == datatype::continuous){
			_params.push_back(__generateContinuousParameters(num_categories, separation));
		}else if(type == datatype::categorical){
			_params.push_back(__generateCategoricalParameters(num_categories, separation));
		}else{
			throw 1;
				// FIXME: throw a proper exception
		}

		vector<double> data_column(_num_rows,0);
		size_t this_category;
		for(size_t row = 0; row < _num_rows; ++row){
			this_category = _row_assignments[this_view][row];
			if(type == datatype::continuous){
				data_column[row] = __generateContinuousDatum(_params[col][this_category]);
			}else if(type == datatype::categorical){
				data_column[row] = __generateCategoricalDatum(_params[col][this_category]);
			}else{
				throw 1;
				// FIXME: throw a proper exception
			}
		}
		_data.push_back(data_column);
	}
}


vector<size_t> SyntheticDataGenerator::generateWeightedParition(vector<double> weights, size_t n)
{
	vector<size_t> partition(n,0);

	if(weights.size()==1) return partition;

	for(size_t i = 0; i < n; ++i)
		partition[i] = (i < weights.size()) ? i : _rng.get()->pflip(weights);

	partition = _rng.get()->shuffle(partition);

	return partition;
}


vector<double> SyntheticDataGenerator::logLikelihood(vector<double> data, size_t column)
{
	auto type = _datatypes[column];
	vector<double> logps;

	if(type == datatype::continuous){
		for(auto x : data)
			logps.push_back(__continuousLogp(x, column));
	}else if(type == datatype::categorical){
		for(auto x : data)
			logps.push_back(__categoricalLogp(x, column));
	}else{
		throw 1; // FIXME: proper error
	}
	return logps;
}

// generator interface
// ````````````````````````````````````````````````````````````````````````````````````````````````
vector<map<string, double>> SyntheticDataGenerator::__generateSeparatedParameters(datatype type,
	size_t num_categories, double separation)
{
	vector<map<string, double>> param_set;

	if(type == datatype::continuous){
		param_set = __generateContinuousParameters(num_categories, separation);
	}else if(type == datatype::categorical){
		param_set = __generateCategoricalParameters(num_categories, separation);
	}else{
		throw 1; // FIXME: proper error
	}
	return param_set;
}


// continuous
// ````````````````````````````````````````````````````````````````````````````````````````````````
vector<map<string, double>> SyntheticDataGenerator::__generateContinuousParameters(
	size_t num_categories, double separation)
{
	// currently assumes homogeneity of variance
	double std = 1;
	double mean_distance = 6*std*separation;

	vector<map<string, double>> params;

	for(size_t i = 0; i < num_categories; ++i){
		map<string, double> param;
		param["rho"] = 1/(std*std);
		param["mu"] = double(i)*mean_distance;
		params.push_back(param);
	}
	return params;
}


double SyntheticDataGenerator::__generateContinuousDatum(map<string, double> param)
{
	// normrand takes standard deviation instead of precision
	return _rng.get()->normrand(param["mu"], 1.0/sqrt(param["rho"]));
}


double SyntheticDataGenerator::__continuousLogp(double x, size_t column)
{
	size_t this_view = _column_assignment[column];
	auto weights = _category_weights[this_view];
	size_t num_clusters = weights.size();

	for(size_t k = 0; k < num_clusters; ++k){
		weights[k] = log(weights[k]) + dist::gaussian::logPdf(x, _params[column][k]["mu"],
			_params[column][k]["rho"]);
	}

	return numerics::logsumexp(weights);
}


// categorical
// ````````````````````````````````````````````````````````````````````````````````````````````````
vector<map<string, double>> SyntheticDataGenerator::__generateCategoricalParameters(
	size_t num_categories, double separation)
{

	if (separation > 0.95)
        separation = 0.95;

	// TODO: Non-arbitrary number of categories.
	size_t K = 5;

	// symmetric
	vector<double> alpha_vec(K, 1.0-separation);

	ASSERT_EQUAL(std::cout, alpha_vec.size(), K);

	vector<map<string, double>> params;

	for(size_t i = 0; i < num_categories; ++i){
		// draw categorical vector for dirichlet
		vector<double> p = _rng.get()->dirrand(alpha_vec);

		ASSERT_EQUAL(std::cout, p.size(), K);
		ASSERT_EQUAL(std::cout, p.size(), alpha_vec.size());

		map<string, double> param;
		for(size_t k = 0; k < K; ++k){
			std::ostringstream key;
			key << k;
			param[key.str()] = p[k];
		}
		params.push_back(param);
	}
	return params;
}


double SyntheticDataGenerator::__generateCategoricalDatum(map<string, double> param)
{
	size_t K = param.size();

	vector<double> p(K);
	for(size_t i = 0; i < K; ++i){
		string key = std::to_string(i);
		p[i] = param[key];
	}

	return static_cast<double>(_rng.get()->pflip(p));
}


double SyntheticDataGenerator::__categoricalLogp(double x, size_t column)
{
	size_t this_view = _column_assignment[column];
	auto weights = _category_weights[this_view];
	size_t num_clusters = weights.size();

	size_t y = static_cast<size_t>(x+.5);

	for(size_t i = 0; i < num_clusters; ++i){
		auto K = _params[column][i].size();
		vector<double> p(K);
		for(size_t k = 0; k < K; ++k){
			string key = std::to_string(k);
			p[k] = _params[column][i][key];
		}
		weights[i] = log(weights[i]) + dist::categorical::logPdf(y, p);
	}

	return numerics::logsumexp(weights);
}


// getters
// ````````````````````````````````````````````````````````````````````````````````````````````````
vector<vector<double>> SyntheticDataGenerator::getData()
{
	return _data;
}

}// end baxcat

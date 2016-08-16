
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

#ifndef baxcat_cxx_synthetic_data_generator_guard
#define baxcat_cxx_synthetic_data_generator_guard

#include <vector>
#include <memory>
#include <sstream>
#include <string>
#include <map>

#include "prng.hpp"
#include "distributions/gaussian.hpp"
#include "distributions/categorical.hpp"
#include "helpers/constants.hpp"
#include "helpers/state_helper.hpp"

namespace baxcat{

// SyntheticDataGenerator generates random crosscat data tables and distributions. It also scores
// data under each column's mixture distribution.
class SyntheticDataGenerator{
public:
	// Quick data
	SyntheticDataGenerator(size_t num_rows, std::vector<std::string> datatypes,
						   unsigned int rng_seed);

	// TODO: implement
	// Generate specific partition for benchmarking
	// SyntheticDataGenerator(size_t num_rows, size_t num_cols, size_t num_views,
	// 	size_t num_clusters_per_view, unsigned int rng_seed);

	// get something specific for inference testing
	SyntheticDataGenerator(size_t num_rows, std::vector<double> view_weights,
						   std::vector<std::vector<double>> category_weights,
						   std::vector<double> category_separation,
						   std::vector<std::string> datatypes, unsigned int rng_seed);

	void initialize();

	std::vector<size_t> generateWeightedParition(std::vector<double> wights, size_t n);

	// get the log likelihood of the data in x given the mixture distribution in column
	std::vector<double> logLikelihood(std::vector<double> data, size_t column);

	// returns the data as a vector of column vectors of doubles---the same format as state takes
	std::vector<std::vector<double>> getData();

	// TODO: implement
	// return distribution parameters for each cluster in a column
	// std::vector<std::map<std::string, double>> getColumnParams(size_t column_index);

	// TODO: implement
	// get assignment of columns to views
	// std::vector<size_t> getColumnAssignment();

	// TODO: implement
	// get assignment of rows to clusters in each view
	// std::vector<std::vector<size_t>> getRowAssignment();

private:
	std::vector<std::map<std::string, double>> __generateSeparatedParameters(baxcat::datatype type,
																			 size_t num_categories,
																			 double separation);

	// generate separated clusters for each type
	std::vector<std::map<std::string, double>> __generateContinuousParameters(size_t num_categories,
																			  double separation);
	std::vector<std::map<std::string, double>> __generateCategoricalParameters(size_t num_categories,
																			   double separation);

	// generate a datum from a set of parameters
	double __generateContinuousDatum(std::map<std::string, double> params);
	double __generateCategoricalDatum(std::map<std::string, double> params);

	// get log lieklihood for the data in X
	double __continuousLogp(double x, size_t column);
	double __categoricalLogp(double x, size_t column);

	// MEMBERS
	size_t _num_rows;
	size_t _num_columns;
	std::vector<size_t> _column_assignment;
	std::vector<std::vector<size_t>> _row_assignments;
	std::vector<std::vector<double>> _category_weights;
	std::vector<std::vector<std::map<std::string, double>>> _params;
	std::vector<double> _view_weights;
	std::vector<double> _category_separation;
	std::vector<baxcat::datatype> _datatypes;
	std::vector<std::vector<double>> _data;

	std::shared_ptr<baxcat::PRNG> _rng;

};

} // end namespace

#endif

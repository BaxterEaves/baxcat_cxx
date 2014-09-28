
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
#include <string>
#include <vector>
#include <iostream>

#include "prng.hpp"
#include "feature.hpp"
#include "helpers/feature_builder.hpp"

using baxcat::BaseFeature;

BOOST_AUTO_TEST_SUITE (feature_builder_test)

BOOST_AUTO_TEST_CASE(construct_single_continuous_feature)
{
    std::vector<std::vector<double>> data_in = {{-2.5, -1, 0, 1, 2.5}};
    std::vector<std::string> datatypes = {"continuous"};
    std::vector<std::vector<double>> distargs = {{}};
    baxcat::PRNG *rng = new baxcat::PRNG;

    std::vector<std::shared_ptr<BaseFeature>> features_out;

    features_out = baxcat::helpers::genFeatures(data_in, datatypes, distargs, rng);

    BOOST_CHECK_EQUAL(features_out[0].get()->getN(), 5);
    BOOST_CHECK_EQUAL(features_out[0].get()->getHypers().size(), 4);
    BOOST_CHECK_EQUAL(features_out[0].get()->getData().size(), 5);
    BOOST_CHECK_EQUAL(features_out[0].get()->getIndex(), 0);

    delete rng;
}

BOOST_AUTO_TEST_CASE(construct_single_categorical_feature)
{
    std::vector<std::vector<double>> data_in = {{0, 1, 2, 3, 4}};
    std::vector<std::string> datatypes = {"categorical"};
    std::vector<std::vector<double>> distargs = {{6}};
    baxcat::PRNG *rng = new baxcat::PRNG;

    std::vector<std::shared_ptr<BaseFeature>> features_out;

    features_out = baxcat::helpers::genFeatures(data_in, datatypes, distargs, rng);

    BOOST_CHECK_EQUAL(features_out[0].get()->getN(), 5);
    BOOST_CHECK_EQUAL(features_out[0].get()->getHypers().size(), 1);
    BOOST_CHECK_EQUAL(features_out[0].get()->getData().size(), 5);
    BOOST_CHECK_EQUAL(features_out[0].get()->getIndex(), 0);

    delete rng;
}

BOOST_AUTO_TEST_CASE(construct_single_feature_of_each_type)
{
    std::vector<std::vector<double>> data_in = {{-2.5, -1, 0, 1, 2.5},{0, 1, 2, 3}};
    std::vector<std::string> datatypes = {"continuous","categorical"};
    std::vector<std::vector<double>> distargs = {{},{6}};
    baxcat::PRNG *rng = new baxcat::PRNG;

    std::vector<std::shared_ptr<BaseFeature>> features_out;

    features_out = baxcat::helpers::genFeatures(data_in, datatypes, distargs, rng);

    BOOST_CHECK_EQUAL(features_out[0].get()->getN(), 5);
    BOOST_CHECK_EQUAL(features_out[0].get()->getHypers().size(), 4);
    BOOST_CHECK_EQUAL(features_out[0].get()->getData().size(), 5);
    BOOST_CHECK_EQUAL(features_out[0].get()->getIndex(), 0);

    BOOST_CHECK_EQUAL(features_out[1].get()->getN(), 4);
    BOOST_CHECK_EQUAL(features_out[1].get()->getHypers().size(), 1);
    BOOST_CHECK_EQUAL(features_out[1].get()->getData().size(), 4);
    BOOST_CHECK_EQUAL(features_out[1].get()->getIndex(), 1);

    delete rng;
}

BOOST_AUTO_TEST_SUITE_END()

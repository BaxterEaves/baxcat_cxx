
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

#include "test_utils.hpp"

using std::vector;
using std::string;


BOOST_AUTO_TEST_SUITE( continuous_inference_single )

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_high_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,.9,
        "continuous", "results/continuous_1col_3clu_highsep.png");

    BOOST_CHECK(!distributions_differ);
}

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_medium_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,.65,
        "continuous", "results/continuous_1col_3clu_midsep.png");

    BOOST_CHECK(!distributions_differ);
}

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_low_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,.4,
        "continuous", "results/continuous_1col_3clu_lowsep.png");

    BOOST_CHECK(!distributions_differ);
}

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_no_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,0.0,
        "continuous", "results/continuous_1col_3clu_nosep.png");

    BOOST_CHECK(!distributions_differ);
}


BOOST_AUTO_TEST_SUITE_END()


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
#include "helpers/state_helper.hpp"
#include "helpers/constants.hpp"


BOOST_AUTO_TEST_SUITE(test_state_helper)

using std::vector;
using std::string;


BOOST_AUTO_TEST_CASE(string_to_transition_should_create_transition)
{

    auto output = baxcat::helpers::string_to_transition.at("row_assignment");
    BOOST_CHECK_EQUAL( output, baxcat::transition_type::row_assignment );

    output = baxcat::helpers::string_to_transition.at("column_assignment");
    BOOST_CHECK_EQUAL( output, baxcat::transition_type::column_assignment );

    output = baxcat::helpers::string_to_transition.at("row_alpha");
    BOOST_CHECK_EQUAL( output, baxcat::transition_type::row_alpha );

    output = baxcat::helpers::string_to_transition.at("column_alpha");
    BOOST_CHECK_EQUAL( output, baxcat::transition_type::column_alpha );

    output = baxcat::helpers::string_to_transition.at("column_hypers");
    BOOST_CHECK_EQUAL( output, baxcat::transition_type::column_hypers );
}

BOOST_AUTO_TEST_CASE(string_to_datatype_should_create_datatype)
{
    auto output = baxcat::helpers::string_to_datatype.at("continuous");
    BOOST_CHECK_EQUAL( output, baxcat::datatype::continuous );
}

BOOST_AUTO_TEST_CASE(get_transitions_should_return_transitions_vector)
{
    vector<string> tstr = {
        "row_assignment",
        "column_assignment",
        "row_alpha",
        "column_alpha",
        "column_hypers"
    };
    auto output = baxcat::helpers::getTransitions( tstr );

    BOOST_REQUIRE_EQUAL( tstr.size(), 5);
    BOOST_REQUIRE_EQUAL( output.size(), tstr.size() );

    BOOST_CHECK_EQUAL( output[0], baxcat::transition_type::row_assignment );
    BOOST_CHECK_EQUAL( output[1], baxcat::transition_type::column_assignment );
    BOOST_CHECK_EQUAL( output[2], baxcat::transition_type::row_alpha );
    BOOST_CHECK_EQUAL( output[3], baxcat::transition_type::column_alpha );
    BOOST_CHECK_EQUAL( output[4], baxcat::transition_type::column_hypers );

}

BOOST_AUTO_TEST_CASE(get_modeltypes_should_return_datatypes_vector)
{
    vector<string> dstr = {
        "continuous",
        "categorical",
        "binomial",
        "count",
        "cyclic",
        "magnitude",
        "bounded"
    };

    auto output = baxcat::helpers::getDatatypes( dstr );

    BOOST_REQUIRE_EQUAL( dstr.size(), 7);
    BOOST_REQUIRE_EQUAL( output.size(), dstr.size() );

    BOOST_CHECK_EQUAL( output[0], baxcat::datatype::continuous );
    BOOST_CHECK_EQUAL( output[1], baxcat::datatype::categorical );
    BOOST_CHECK_EQUAL( output[2], baxcat::datatype::binomial );
    BOOST_CHECK_EQUAL( output[3], baxcat::datatype::count );
    BOOST_CHECK_EQUAL( output[4], baxcat::datatype::cyclic );
    BOOST_CHECK_EQUAL( output[5], baxcat::datatype::magnitude );
    BOOST_CHECK_EQUAL( output[6], baxcat::datatype::bounded );
}
BOOST_AUTO_TEST_SUITE_END()


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

#include "state.hpp"
#include "test_utils.hpp"
#include "utils.hpp"


BOOST_AUTO_TEST_SUITE (test_state)

using std::vector;
using std::map;
using std::string;
using std::shared_ptr;
using std::cout;
using std::endl;

using baxcat::State;
using baxcat::test_utils::areIdentical;

struct Setup
{
    unsigned int seed = 10;
    vector<vector<double>> data = {
        {0.5377, 1.8339, -2.2588, 0.8622, 0.3188},
        {-1.3077, -0.4336, 0.3426, 3.5784, 2.7694}
    };
    vector<string> datatypes = {"continuous", "continuous"};
    vector<size_t> column_assignment = {0,1};
    vector<vector<size_t>> row_assignments = {{0,0,0,0,0},{0,1,2,3,4}};
    vector<vector<double>> distargs = {{0},{0}};
};

// Constructors
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(verify_constructor_1){
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed);

    auto column_assignment = state.getColumnAssignment();
    auto row_assignments = state.getRowAssignments();

    // column_assignment should have an entry for each column
    BOOST_REQUIRE_EQUAL(column_assignment.size(), 2);

    // each vector in row_assignments should have an entry for each row
    BOOST_REQUIRE_EQUAL(row_assignments[0].size(), 5);

    auto view_alphas = state.getViewCRPAlphas();

    // view_alphas should be the same length as row_assignments
    BOOST_REQUIRE_EQUAL( view_alphas.size(), row_assignments.size());

    auto col_hypers = state.getColumnHypers();
    BOOST_REQUIRE_EQUAL(col_hypers.size(), 2);

}

BOOST_AUTO_TEST_CASE(verify_constructor_2){
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto column_assignment = state.getColumnAssignment();
    auto row_assignments = state.getRowAssignments();

    // column_assignment should have an entry for each column
    BOOST_REQUIRE_EQUAL(column_assignment.size(), 2);

    // each vector in row_assignments should have an entry for each row
    BOOST_REQUIRE_EQUAL(row_assignments[0].size(), 5);

    auto view_alphas = state.getViewCRPAlphas();

    // view_alphas should be the same length as row_assignments
    BOOST_REQUIRE_EQUAL(view_alphas.size(), row_assignments.size());

    auto col_hypers = state.getColumnHypers();
    BOOST_REQUIRE_EQUAL(col_hypers.size(), 2);

    BOOST_REQUIRE_EQUAL(state.getNumViews(), 2);

    // check that all the things are what I set them to
    // column_assignment
    BOOST_CHECK(areIdentical(s.column_assignment, column_assignment));
    // row_assignments
    BOOST_REQUIRE_EQUAL(row_assignments.size(), s.row_assignments.size() );
    BOOST_REQUIRE_EQUAL(row_assignments[0].size(), s.row_assignments[0].size());
    BOOST_REQUIRE_EQUAL(row_assignments[1].size(), s.row_assignments[1].size());

    BOOST_CHECK_EQUAL(areIdentical(row_assignments[0], s.row_assignments[0]), 1);
    BOOST_CHECK_EQUAL(areIdentical(row_assignments[1], s.row_assignments[1]), 1);

}
// Transitions
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(transition_col_alpha_should_change_from_out_range_val)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto alpha_0 = state.getStateCRPAlpha();
    BOOST_CHECK(alpha_0 > 0);

    state.transition({"column_alpha"}, {}, {}, 0, 1);

    auto alpha_1 = state.getStateCRPAlpha();

    BOOST_CHECK(alpha_0 != alpha_1);
    BOOST_CHECK(alpha_1 > 0);
}

BOOST_AUTO_TEST_CASE(transition_row_alpha_should_change_vals_eventually)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto view_alphas_0 = state.getViewCRPAlphas();
    for( auto a : view_alphas_0)
        BOOST_CHECK( a > 0);

    state.transition({"row_alpha"}, {}, {}, 0, 1);

    auto view_alphas_1 = state.getViewCRPAlphas();
    for( auto a : view_alphas_1)
        BOOST_CHECK( a > 0);

    bool values_changed = false;
    for( int i = 0; i < 100; i++){
        state.transition({"row_alpha"}, {}, {}, 0, 1);
        auto view_alphas_i = state.getViewCRPAlphas();
        if( areIdentical(view_alphas_0, view_alphas_i) != 1 ){
            values_changed = true;
            break;
        }
    }

    BOOST_CHECK( values_changed );
}

BOOST_AUTO_TEST_CASE(row_z_should_eventually_change_assignments)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto row_assignments_0 = state.getRowAssignments();

    bool partition_change = false;
    for( int i = 0; i < 100; i++){
        state.transition({"row_assignment"}, {}, {}, 0, 1);
        auto row_assignments_i = state.getRowAssignments();
        // paritions should be the same size
        BOOST_REQUIRE_EQUAL( row_assignments_i.size(), row_assignments_0.size());
        for(size_t j = 0; j < row_assignments_i.size(); ++j){
            // subparitions should be the same size
            BOOST_REQUIRE_EQUAL( row_assignments_i[j].size(), row_assignments_0[j].size());
            if( areIdentical(row_assignments_0[j], row_assignments_i[j]) != 1 );
            partition_change = true;
            break;
        }
        if(partition_change) break;
    }
    BOOST_CHECK( partition_change );
}

BOOST_AUTO_TEST_CASE(col_hypers_should_eventually_change)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto hypers_0 = state.getColumnHypers();

    bool hypers_change = false;
    for( int i = 0; i < 100; i++){
        state.transition({"column_hypers"}, {}, {}, 0, 1);
        auto hypers_i = state.getColumnHypers();
        for( int col = 0; col < 2; ++col){
            if( hypers_i[col]["m"] != hypers_0[col]["m"] ||
                hypers_i[col]["r"] != hypers_0[col]["r"] ||
                hypers_i[col]["s"] != hypers_0[col]["s"] ||
                hypers_i[col]["nu"] != hypers_0[col]["nu"])
            {
                hypers_change = true;
                break;
            }
        }
        if(hypers_change) break;
    }

    BOOST_CHECK( hypers_change );
}

BOOST_AUTO_TEST_CASE(col_partition_should_eventually_change)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto column_assignment_0 = state.getColumnAssignment();
    bool partition_changed = false;
    for(int i = 0; i < 100; i++){
        state.transition({"column_assignment"}, {}, {}, 0, 1);
        auto column_assignment_i = state.getColumnAssignment();
        if(not areIdentical(column_assignment_0, column_assignment_i) ){
            partition_changed = true;
            break;
        }
    }
    BOOST_CHECK( partition_changed );

}

// geweke functions
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(geweke_pullDataColumn_value_checks)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment, s.row_assignments, {});

    auto data_f0 = state.__geweke_pullDataColumn(0);
    BOOST_CHECK(areIdentical(s.data[0], data_f0));

    auto data_f1 = state.__geweke_pullDataColumn(1);
    BOOST_CHECK(areIdentical(s.data[1], data_f1));
}

// quick datatype crash tests
// `````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(categorical_crash_test)
{
    unsigned int seed = 10;
    vector<vector<double>> data = {
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4}
    };
    vector<string> datatypes = {"categorical", "categorical"};
    vector<size_t> column_assignment = {0, 1};
    vector<vector<size_t>> row_assignments = {{0, 0, 0,0 ,0}, {0, 1, 2, 3, 4}};
    vector<vector<double>> distargs = {{5}, {6}};

    State state(data, datatypes, distargs, seed, column_assignment, row_assignments, {});

    auto column_assignments_out = state.getColumnAssignment();
    auto row_assignments_out = state.getRowAssignments();

    // column_assignment should have an entry for each column
    BOOST_REQUIRE_EQUAL(column_assignments_out.size(), 2);

    // each vector in row_assignments should have an entry for each row
    BOOST_REQUIRE_EQUAL(row_assignments_out[0].size(), 5);

    auto view_alphas = state.getViewCRPAlphas();

    // view_alphas should be the same length as row_assignments
    BOOST_REQUIRE_EQUAL(view_alphas.size(), row_assignments_out.size());

    auto col_hypers = state.getColumnHypers();
    BOOST_REQUIRE_EQUAL(col_hypers.size(), 2);

    // column_assignment
    BOOST_CHECK(areIdentical(column_assignment, column_assignment));
    // row_assignments
    BOOST_REQUIRE_EQUAL(row_assignments.size(), row_assignments_out.size() );
    BOOST_REQUIRE_EQUAL(row_assignments[0].size(), row_assignments_out[0].size());
    BOOST_REQUIRE_EQUAL(row_assignments[1].size(), row_assignments_out[1].size());

    BOOST_CHECK_EQUAL(areIdentical(row_assignments[0], row_assignments_out[0]), 1);
    BOOST_CHECK_EQUAL(areIdentical(row_assignments[1], row_assignments_out[1]), 1);

}
BOOST_AUTO_TEST_SUITE_END ()

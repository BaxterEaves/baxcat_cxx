
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

const double EPSILON = 10E-10;

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
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

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
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

    auto alpha_0 = state.getStateCRPAlpha();
    BOOST_CHECK(alpha_0 > 0);

    state.transition({"column_alpha"}, vector<size_t>(), vector<size_t>(), 0, 1);

    auto alpha_1 = state.getStateCRPAlpha();

    BOOST_CHECK(alpha_0 != alpha_1);
    BOOST_CHECK(alpha_1 > 0);
}

BOOST_AUTO_TEST_CASE(transition_row_alpha_should_change_vals_eventually)
{
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

    auto view_alphas_0 = state.getViewCRPAlphas();
    for( auto a : view_alphas_0)
        BOOST_CHECK( a > 0);

    state.transition({"row_alpha"}, vector<size_t>(), vector<size_t>(), 0, 1);

    auto view_alphas_1 = state.getViewCRPAlphas();
    for( auto a : view_alphas_1)
        BOOST_CHECK( a > 0);

    bool values_changed = false;
    for( int i = 0; i < 100; i++){
        state.transition({"row_alpha"}, vector<size_t>(), vector<size_t>(), 0, 1);
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
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

    auto row_assignments_0 = state.getRowAssignments();

    bool partition_change = false;
    for( int i = 0; i < 100; i++){
        state.transition({"row_assignment"}, vector<size_t>(), vector<size_t>(), 0, 1);
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
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

    auto hypers_0 = state.getColumnHypers();

    bool hypers_change = false;
    for( int i = 0; i < 100; i++){
        state.transition({"column_hypers"}, vector<size_t>(), vector<size_t>(), 0, 1);
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
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

    auto column_assignment_0 = state.getColumnAssignment();
    bool partition_changed = false;
    for(int i = 0; i < 100; i++){
        state.transition({"column_assignment"}, vector<size_t>(), vector<size_t>(), 0, 1);
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
    State state(s.data, s.datatypes, s.distargs, s.seed, s.column_assignment,
                s.row_assignments, -1, vector<double>(), vector<map<string, double>>());

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

    State state(data, datatypes, distargs, seed, column_assignment, row_assignments,
                -1, vector<double>(), vector<map<string, double>>());

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

// get data row and table
//``````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(get_row_should_return_data){
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed);

    auto row_data = state.getDataRow(0);
    BOOST_CHECK_EQUAL(row_data.size(), 2);
    BOOST_CHECK_CLOSE_FRACTION(row_data[0], 0.5377, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(row_data[1], -1.3077, EPSILON);

    row_data = state.getDataRow(3);
    BOOST_CHECK_EQUAL(row_data.size(), 2);
    BOOST_CHECK_CLOSE_FRACTION(row_data[0], 0.8622, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(row_data[1], 3.5784, EPSILON);
}

BOOST_AUTO_TEST_CASE(get_table_should_return_data){
    Setup s;
    State state(s.data, s.datatypes, s.distargs, s.seed);

    auto data = state.getDataTable();
    BOOST_CHECK_EQUAL(data.size(), 5);
    BOOST_CHECK_EQUAL(data[0].size(), 2);
    BOOST_CHECK_CLOSE_FRACTION(data[0][0], 0.5377, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(data[0][1], -1.3077, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(data[3][0], 0.8622, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(data[3][1], 3.5784, EPSILON);
}

// replace data (slice and row) tests
//``````````````````````````````````````````````````````````````````````````````````````````````````
// BOOST_AUTO_TEST_CASE(replace_slice_data_should_update_suffstats){
//     // Fixme: implement!
//     BOOST_CHECK(false);
// }
// 
// BOOST_AUTO_TEST_CASE(replace_row_data_should_update_suffstats){
//     // Fixme: implement!
//     BOOST_CHECK(false);
// }

BOOST_AUTO_TEST_SUITE_END ()

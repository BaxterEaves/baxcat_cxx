
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

#ifndef baxcat_cxx_state_helper_guard
#define baxcat_cxx_state_helper_guard

#include <iostream>
#include <vector>
#include <string>

#include "helpers/constants.hpp"

namespace baxcat { namespace helpers{


const std::vector<transition_type> all_transitions
{
    row_assignment,
    column_assignment,
    row_alpha,
    column_alpha,
    column_hypers
};


// convert a datatype string from python to a model name
const std::map<std::string, datatype> string_to_datatype
{
    {"continuous", datatype::continuous},
    {"categorical", datatype::categorical},
    {"binomial", datatype::binomial},
    {"count", datatype::count},
    {"cyclic", datatype::cyclic},
    {"magnitude", datatype::magnitude},
    {"bounded", datatype::bounded}
};


const std::map<datatype, bool> type_to_is_discrete
{
    {datatype::continuous, false},
    {datatype::categorical, true},
    {datatype::binomial, true},
    {datatype::count, true},
    {datatype::magnitude, false},
    {datatype::cyclic, false},
    {datatype::bounded, false},
};


// convert a string from python to a transition
const std::map<std::string, transition_type> string_to_transition
{
    {"row_assignment", transition_type::row_assignment},
    {"column_assignment", transition_type::column_assignment},
    {"row_alpha", transition_type::row_alpha},
    {"column_alpha", transition_type::column_alpha},
    {"column_hypers", transition_type::column_hypers}
};


// converts a list of transition string into a list of transitions
static std::vector<transition_type> getTransitions(std::vector< std::string > tstr)
{
    std::vector<transition_type> ret;
    for(auto s : tstr)
        ret.push_back( string_to_transition.at(s) );
    return ret;
};


// converts a list of transition string into a list of transitions
static std::vector<datatype> getDatatypes(std::vector< std::string > dstr)
{
    std::vector<datatype> ret;
    for(auto s : dstr)
        ret.push_back(string_to_datatype.at(s));
    return ret;
};


static bool is_discrete(datatype type)
{
    return type_to_is_discrete.at(type);
};

}} // end namespaces

#endif

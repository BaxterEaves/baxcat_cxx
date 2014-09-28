
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

#ifndef baxcat_cxx_distributions_template_helpers_hpp
#define baxcat_cxx_distributions_template_helpers_hpp


namespace baxcat{

    template <typename Condition, typename T = void>
    using enable_if = typename std::enable_if<Condition::value, T>::type;

    template <typename Condition, typename T = void>
    using disable_if = typename std::enable_if<!Condition::value, T>::type;
}

#endif

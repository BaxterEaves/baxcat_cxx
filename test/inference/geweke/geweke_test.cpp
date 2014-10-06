
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

#include "geweke_tester.hpp"
#include <vector>
#include <string>

int main(){

    // std::vector<std::string> datatypes = {"categorical", "continuous", "categorical",
    //                                       "continuous","categorical"};
    std::vector<std::string> datatypes = {"categorical"};

	baxcat::GewekeTester gwk(20, 1, datatypes, 84715, true, true, true);

    size_t lag = 5;

	gwk.run(100000, 5, lag);

	gwk.outputResults();

	return 0;
}

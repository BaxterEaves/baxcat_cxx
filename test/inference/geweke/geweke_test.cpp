
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

#include "boost/program_options.hpp"

int main(int argc, char** argv)
{
    // default values
    size_t iters, chains, rows, seed, lag, m, ct_kernel;
    bool fix_hypers, fix_row_alpha, fix_col_alpha, fix_row_z, fix_col_z;
    
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options() 
        ("help", "Print help messages")
        ("fix_hypers,y", po::bool_switch(&fix_hypers)->default_value(false), "fix hyperparamters transition")
        ("fix_row_alpha,a", po::bool_switch(&fix_row_alpha)->default_value(false), "fix row in views CRP alpha transition")
        ("fix_col_alpha,v", po::bool_switch(&fix_col_alpha)->default_value(false), "fix cols in state CRP alpha transition")
        ("fix_row_z,w", po::bool_switch(&fix_row_z)->default_value(false), "fix row assignment")
        ("fix_col_z,z", po::bool_switch(&fix_col_z)->default_value(false), "fix column assignment")
        ("iters,i", po::value<size_t>(&iters)->default_value(10000), "number of iterations per chain")
        ("chains,h", po::value<size_t>(&chains)->default_value(5), "number of chains")
        ("rows,r", po::value<size_t>(&rows)->default_value(10), "number of rows")
        ("ct_kernel,k", po::value<size_t>(&ct_kernel)->default_value(0), "column transition kernel")
        (",m", po::value<size_t>(&m)->default_value(1), "view transition kernel aux paramter")
        ("seed,s", po::value<size_t>(&seed)->default_value(1281990), "random number generator seed")
        ("lag,l", po::value<size_t>(&lag)->default_value(5), "number of iterations between sample collection");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc),  vm); // can throw 

    
    // if (vm.count("help")){ 
    //     std::cout << "Basic Command Line Parameter App" << std::endl << desc << std::endl;
    //     return 0; 
    // } 

    po::notify(vm);

    std::cout << "=================================================" << std::endl;
    std::cout << "fix col alpha: " << fix_col_alpha << std::endl
              << "fix row alpha: " << fix_row_alpha << std::endl
              << "fix col z: " << fix_col_z << std::endl
              << "fix row z: " << fix_row_z << std::endl
              << "fix hyperparamters: " << fix_hypers << std::endl
              << "seed: " << seed << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "rows: " << rows << std::endl
              << "iters: " << iters << std::endl
              << "chains: " << chains << std::endl
              << "lag: " << lag << std::endl;
    std::cout << "=================================================" << std::endl;

    std::vector<std::string> datatypes = {"categorical", "continuous", "categorical", 
                                          "continuous", "categorical", "categorical", 
                                          "continuous", "categorical", "continuous", 
                                          "categorical"};

    // std::vector<std::string> datatypes = {"continuous"};

	baxcat::GewekeTester gwk(rows, datatypes.size(), datatypes, seed, m, !fix_hypers, !fix_row_alpha, 
                             !fix_col_alpha, !fix_row_z, !fix_col_z, ct_kernel);

	gwk.run(iters, chains, lag);

	gwk.outputResults();

	return 0;
}

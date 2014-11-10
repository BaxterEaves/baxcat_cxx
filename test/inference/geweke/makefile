 
# BaxCat: an extensible cross-catigorization engine.
# Copyright (C) 2014 Baxter Eaves
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License (LICENSE.txt) along with this
# program. If not, see <http://www.gnu.org/licenses/>.
#
# You may contact the mantainers of this software via github
# <https://github.com/BaxterEaves/baxcat_cxx>.

# TODO: Learn how to make better makefiles

SRC = ../../../src
INCLUDE = ../../../include
BOOST_DIR = /usr/local/Cellar/boost/1.56.0/include
BOOST_LIB = /usr/local/Cellar/boost/1.56.0/lib

# output folder
F_OUT = build/

CXX = g++-4.9
OPTS = -I $(BOOST_DIR) -I $(INCLUDE) -std=c++11 -fopenmp -I /usr/local/Cellar/mathgl/2.2.2.1/include \
	-I ../../ -L /usr/local/Cellar/mathgl/2.2.2.1/lib -L $(BOOST_LIB) -lboost_program_options-mt \
	-lmgl -Wall -Wextra -Wunreachable-code \
	-Wno-unused-function -Wno-unused-local-typedefs -O3 -DDEBUG -Wno-comment

TEST_FILES = geweke.inftest

# test objects
#``````````````````````````````````````````````````````````````````````````````````````````````````
all: $(TEST_FILES)

geweke.inftest:
	$(CXX) geweke_test.cpp -o $(F_OUT)geweke.inftest \
	$(SRC)/continuous.cpp $(SRC)/categorical.cpp $(SRC)/synthetic_data_generator.cpp \
	$(SRC)/state.cpp $(SRC)/view.cpp $(SRC)/feature_tree.cpp \
	geweke_tester.cpp $(OPTS)

runtests: $(TEST_FILES)
	./$(F_OUT)geweke.inftest;

clean:
	rm *.o; rm build/*.inftest

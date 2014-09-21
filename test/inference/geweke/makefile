# TODO: Learn how to make better makefiles

SRC = ../../../src
INCLUDE = ../../../include
BOOST_DIR = /usr/local/include/boost_1_55_0

# output folder
F_OUT = build/

CXX = g++-4.9
OPTS = -I$(BOOST_DIR) -I $(INCLUDE) -std=c++11 -fopenmp -I /usr/local/Cellar/mathgl/2.2.2.1/include \
	-I ../../ -L /usr/local/Cellar/mathgl/2.2.2.1/lib -lmgl -Wall -Wextra -Wunreachable-code\
	-Wno-unused-function -Wno-unused-local-typedefs -O3 -DDEBUG

TEST_FILES = geweke.inftest

# test objects
#``````````````````````````````````````````````````````````````````````````````````````````````````
all: $(TEST_FILES)

geweke.inftest:
	$(CXX) geweke_test.cpp -o $(F_OUT)geweke.inftest \
	$(SRC)/continuous.cpp $(SRC)/synthetic_data_generator.cpp \
	$(SRC)/state.cpp $(SRC)/view.cpp $(SRC)/feature_tree.cpp \
	geweke_tester.cpp \
	$(OPTS)

runtests: $(TEST_FILES)
	./$(F_OUT)geweke.inftest;

clean:
	rm *.o; rm build/*.inftest

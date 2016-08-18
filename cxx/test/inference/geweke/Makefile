
# TODO: Learn how to make better makefiles

SRC = ../../../src
INCLUDE = ../../../include

# output folder
F_OUT = build/

OPTS = -I $(INCLUDE) -std=c++11 -fopenmp -I /usr/local/Cellar/mathgl/2.2.2.1/include \
	-I ../../ -L /usr/local/Cellar/mathgl/2.2.2.1/lib -lboost_program_options \
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
	$(SRC)/geweke_tester.cpp $(OPTS)

runtests: $(TEST_FILES)
	./$(F_OUT)geweke.inftest;

clean:
	rm *.o; rm build/*.inftest

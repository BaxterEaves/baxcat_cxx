# TODO: Learn how to make better makefiles

SRC = ../../src
INCLUDE = ../../include
BOOST_DIR = /usr/local/Cellar/boost/1.55.0_2/include
BOOST_LIB = /usr/local/Cellar/boost/1.55.0_2/lib
DIST = $(INCLUDE)/distributions
HELPERS = $(INCLUDE)/helpers

CXX = g++-4.9
OPTS = -I$(BOOST_DIR) -I$(INCLUDE) \
	-I /usr/local/Cellar/mathgl/2.2.2.1/include \
	-L /usr/local/Cellar/mathgl/2.2.2.1/lib -lmgl \
	-I ../ -std=c++14 -fopenmp \
	-Wall -Wno-unused-function -Wno-unused-local-typedefs \
	-L $(BOOST_LIB) -DDEBUG

DEBUG_HEADER = $(INCLUDE)/debug.hpp
OBJECTS = categorical.o continuous.o state.o view.o feature_tree.o sdg.o
TEST_FILES = utils_test.cpp test_utils_test.cpp numerics_test.cpp container_test.cpp prng_test.cpp \
	distributions_test.cpp nng_test.cpp continuous_test.cpp feature_test.cpp feature_tree_test.cpp \
	view_test.cpp state_test.cpp synthetic_data_generator_test.cpp msd_test.cpp categorical_test.cpp


all : $(TEST_FILES) $(OBJECTS) test_main.cpp
	$(CXX) test_main.cpp $(TEST_FILES) -o main.test $(OPTS) $(OBJECTS)

categorical.o : $(INCLUDE)/component.hpp $(INCLUDE)/utils.hpp $(INCLUDE)/prng.hpp $(DEBUG_HEADER) \
		$(DIST)/gamma.hpp $(DIST)/gaussian.hpp $(DIST)/students_t.hpp $(INCLUDE)/samplers/slice.hpp \
		$(INCLUDE)/models/msd.hpp $(INCLUDE)/datatypes/categorical.hpp $(SRC)/categorical.cpp
	$(CXX) -c $(SRC)/categorical.cpp $(OPTS)

continuous.o : $(INCLUDE)/component.hpp $(INCLUDE)/utils.hpp $(INCLUDE)/prng.hpp $(DEBUG_HEADER) \
		$(SRC)/continuous.cpp $(DIST)/gamma.hpp $(DIST)/gaussian.hpp $(DIST)/students_t.hpp \
		$(DIST)/inverse_gamma.hpp $(INCLUDE)/samplers/slice.hpp $(INCLUDE)/models/nng.hpp
	$(CXX) -c $(SRC)/continuous.cpp $(OPTS)

feature_tree.o : continuous.o $(INCLUDE)/feature.hpp $(INCLUDE)/feature.tpp \
		$(SRC)/feature_tree.cpp $(HELPERS)/feature_tree.hpp
	$(CXX) -c $(SRC)/feature_tree.cpp continuous.o $(OPTS)

view.o : continuous.o feature_tree.o $(INCLUDE)/view.hpp $(SRC)/view.cpp
	$(CXX) -c $(SRC)/view.cpp continuous.o feature_tree.o $(OPTS)

state.o : continuous.o view.o feature_tree.o $(SRC)/state.cpp \
		$(INCLUDE)/state.hpp $(HELPERS)/feature_builder.hpp $(HELPERS)/state_helper.hpp
	$(CXX) -c $(SRC)/state.cpp view.o continuous.o feature_tree.o $(OPTS)

sdg.o : $(HELPERS)/constants.hpp $(HELPERS)/state_helper.hpp $(DIST)/gaussian.hpp $(DIST)/multinomial.hpp \
		$(INCLUDE)/prng.hpp $(HELPERS)/synthetic_data_generator.hpp $(SRC)/synthetic_data_generator.cpp
	$(CXX) -c $(SRC)/synthetic_data_generator.cpp -o sdg.o $(OPTS)

runtests:
	./main.test  # --log_level=test_suite

runjenkins:
	./main.test --log_format=XML --log_sink=results.xml --log_level=all --report_level=no

clean:
	rm *.test; rm *.o;

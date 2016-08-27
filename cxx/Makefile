
SRC := src
INCLUDE := include
UNIT := test/unit
GEWEKE := test/inference/geweke

SRCFILES := $(shell find $(SRC) -name '*.cpp') \
	$(shell find $(UNIT) -name '*test.cpp')
OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))
DEPFILES := $(patsubst %.cpp,%.d,$(SRCFILES))

WARNINGS := -Wall -Wno-unused-function -Wno-unused-local-typedefs \
			-Wno-comment -Wno-reorder
CFLAGS   := -I$(INCLUDE) -std=c++11 -fopenmp -lmgl -DDEBUG $(WARNINGS)


.PHONY: all clean todolist geweke


all: unit

unit: $(OBJFILES)
	@$(CXX) -c $(UNIT)/test_main.cpp -o $(UNIT)/test_main.o $(CFLAGS) -MMD -MP
	@$(CXX) $(OBJFILES) $(UNIT)/test_main.o -o unit.test $(CFLAGS) 
	./unit.test --log_level=error --report_level=short

geweke: 
	$(CXX) $(GEWEKE)/geweke_test.cpp -o geweke.test \
		$(SRC)/continuous.cpp $(SRC)/categorical.cpp \
		$(SRC)/synthetic_data_generator.cpp $(SRC)/state.cpp $(SRC)/view.cpp \
		$(SRC)/feature_tree.cpp $(SRC)/geweke_tester.cpp $(CFLAGS) \
		-lboost_program_options -lmgl -O3
	./geweke.test

clean:
	-@$(RM) $(wildcard $(OBJFILES) $(DEPFILES) *.test)

%.o: %.cpp Makefile
	@$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

todolist:
	-@for file in $(ALLFILES:Makefile=); do fgrep -H -e TODO -e FIXME $$file; \
		done; true

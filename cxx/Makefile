
SRC := src
INCLUDE := include
UNIT := test/unit
GEWEKE := test/inference/geweke

SRCFILES := $(shell find $(SRC) -name '*.cpp') $(shell find $(UNIT) -name '*test.cpp')
OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))
DEPFILES := $(patsubst %.cpp,%.d,$(SRCFILES))

WARNINGS := -Wall -Wno-unused-function -Wno-unused-local-typedefs -Wno-comment
CFLAGS   := -I$(INCLUDE) -std=c++11 -fopenmp -lmgl -DDEBUG $(WARNINGS)


.PHONY: all clean todolist geweke


all : unit

unit : $(OBJFILES)
	@$(CXX) -c $(UNIT)/test_main.cpp -o $(UNIT)/test_main.o $(CFLAGS) -MMD -MP
	@$(CXX) $(OBJFILES) $(UNIT)/test_main.o -o unit.test $(CFLAGS) 
	./unit.test --log_level=error --report_level=short

geweke : $(OBJFILES)
	@$(CXX) -c $(GEWEKE)/geweke_test.cpp -o $(GEWEKE)/geweke_test.o $(CFLAGS) -MMD -MP -lboost_program_options
	@$(CXX) $(OBJFILES)  $(GEWEKE)/geweke_test.o -o geweke.test $(CFLAGS)
	./geweke.test

clean:
	-@$(RM) $(wildcard $(OBJFILES) $(DEPFILES) *.test)

%.o: %.cpp Makefile
	@$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

todolist:
	-@for file in $(ALLFILES:Makefile=); do fgrep -H -e TODO -e FIXME $$file; done; true

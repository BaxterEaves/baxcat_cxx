
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

TARGET = main.test

SRC := ../../src
INCLUDE := ../../include

AUXFILES := Makefile

SRCFILES := $(shell find $(SRC) -name '*.cpp') $(shell find . -name '*.cpp')
HDRFILES := $(shell find $(INCLUDE) -name '*pp')

OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))
TSTFILES := $(patsubst %.cpp,%_t,$(SRCFILES))

DEPFILES    := $(patsubst %.cpp,%.d,$(SRCFILES))
TSTDEPFILES := $(patsubst %,%.d,$(TSTFILES))

ALLFILES := $(SRCFILES) $(HDRFILES) $(AUXFILES)

WARNINGS := -Wall -Wno-unused-function -Wno-unused-local-typedefs -Wno-comment
CFLAGS   := -I$(INCLUDE) -std=c++11 -fopenmp -lmgl -DDEBUG $(WARNINGS)


.PHONY: all clean testmake todolist test

all : $(TARGET)

$(TARGET) : $(OBJFILES)
	@$(CXX) $(CFLAGS) $(OBJFILES) -o $(TARGET)
	./main.test --log_level=test_suite

clean:
	-@$(RM) $(wildcard $(OBJFILES) $(DEPFILES) $(TSTFILES) $(TARGET))

test:
	./main.test --log_level=test_suite

%.o: %.cpp Makefile
	@$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

%_t: %.c Makefile main.test
	@$(CXX) $(CFLAGS) -MMD -MP -DTEST $< main.test -o $@

todolist:
	-@for file in $(ALLFILES:Makefile=); do fgrep -H -e TODO -e FIXME $$file; done; true

testmake:
	@echo OBJFILES = $(OBJFILES)
	@echo SRCFILES = $(SRCFILES)
	@echo DEPFILES = $(DEPFILES)

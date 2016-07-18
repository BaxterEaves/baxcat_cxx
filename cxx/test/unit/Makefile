
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

SRCFILES := $(shell find $(SRC) -name '*.cpp') $(shell find . -name '*.cpp')
OBJFILES := $(patsubst %.cpp,%.o,$(SRCFILES))

WARNINGS := -Wall -Wno-unused-function -Wno-unused-local-typedefs -Wno-comment
CFLAGS   := -I$(INCLUDE) -std=c++11 -fopenmp -lmgl -DDEBUG $(WARNINGS)


.PHONY: all clean todolist

all : $(TARGET)

$(TARGET) : $(OBJFILES)
	@$(CXX) $(CFLAGS) $(OBJFILES) -o $(TARGET)
	./$(TARGET) --log_level=all

clean:
	-@$(RM) $(wildcard $(OBJFILES) $(TARGET))

%.o: %.cpp Makefile
	@$(CXX) $(CFLAGS) -MMD -MP -c $< -o $@

todolist:
	-@for file in $(ALLFILES:Makefile=); do fgrep -H -e TODO -e FIXME $$file; done; true

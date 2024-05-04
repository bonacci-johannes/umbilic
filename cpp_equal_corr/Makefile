# Makefile for compiling program

# Compiler settings
CXX = g++
CXXFLAGS = -O2 -std=c++11

# Source file
SRC = main.cpp

# Executable name
EXEC = program

# Makefile targets
all: $(EXEC)

$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(EXEC)

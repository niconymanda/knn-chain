# Variables
CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -mavx2 $(shell python3 -m pybind11 --includes)
LDFLAGS = $(shell python3-config --ldflags)

# Targets and dependencies
all: knn_chain.so

knn_chain.so: knn_chain.cpp
	$(CXX) $(CXXFLAGS) knn_chain.cpp -o knn_chain$(shell python3-config --extension-suffix) $(LDFLAGS)

clean:
	rm -f knn_chain$(shell python3-config --extension-suffix)

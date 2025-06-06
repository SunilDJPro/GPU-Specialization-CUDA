IDIR=./
STD :=c++20
CXX=g++
CXXFLAGS=-I$(DIR) -Wall -Wextra -pedantic-errors -std=$(STD) -O2 -pthread

build: *.cpp
	$(CXX) -o producer_consumer.exe $(CXXFLAGS) *.cpp

.PHONY: clean

clean:
	rm -f producer_consumer.exe

run:
	./producer_consumer.exe
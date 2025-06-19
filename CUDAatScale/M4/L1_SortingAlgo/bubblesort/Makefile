IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -I../../../lib/cub/ -I../../../lib/cuda-samples/Common -lcudart -lcuda --std c++17

.PHONY: clean build run

build: *.cu
	$(COMPILER) $(COMPILER_FLAGS) *.cu -o generic-bubble-sort.exe

clean:
	rm -f generic-bubble-sort.exe

run:
	./generic-bubble-sort.exe

all: clean build run
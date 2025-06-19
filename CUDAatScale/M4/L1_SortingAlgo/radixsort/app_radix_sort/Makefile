IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -I../../../lib/cub/ -I../../../lib/cuda-samples/Common -lcudart -lcuda --std c++17

.PHONY: clean build run

build: *.cu
	$(COMPILER) $(COMPILER_FLAGS) *.cu -o radix_sort.exe

clean:
	rm -f radix_sort.exe

run:
	./radix_sort.exe

all: clean build run

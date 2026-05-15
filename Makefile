.PHONY: all run clean

APPS := $(patsubst apps/%.cpp,build/%,$(wildcard apps/*.cpp))
SRCS := $(wildcard src/*.cpp) $(wildcard src/*.cu)

all: $(APPS)

build/%: apps/%.cpp $(SRCS) include/*.hpp include/*.cuh | build
	nvcc -std=c++17 -O2 -Iinclude -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89 $< $(SRCS) -o $@ -lcusparse
build:
	mkdir -p build

run: build/validate_implementations
	./build/validate_implementations data/*.mtx

clean:
	rm -rf build

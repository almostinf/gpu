CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
# Match CPU/OpenCL arithmetic when comparing stochastic trajectories.
CXXFLAGS += -ffp-contract=off
ifeq ($(shell uname -s),Darwin)
LIBOMP_PREFIX ?= $(shell brew --prefix libomp)
OPENMP_CXXFLAGS ?= -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
OPENMP_LIBS ?= -L$(LIBOMP_PREFIX)/lib -Wl,-rpath,$(LIBOMP_PREFIX)/lib -lomp
OPENCL_LIB ?= -framework OpenCL
else
OPENMP_CXXFLAGS ?= -fopenmp
OPENMP_LIBS ?= -fopenmp
OPENCL_LIB ?= $(shell pkg-config --libs OpenCL 2>/dev/null || echo -l:libOpenCL.so.1)
endif

# PPM rendering and benchmarking do not depend on the video encoder.
WITH_THEORA ?= $(shell pkg-config --exists theoraenc 2>/dev/null && echo 1 || echo 0)
ifeq ($(WITH_THEORA),1)
VIDEO_SOURCE = theora.cc
VIDEO_FLAGS = -DHAVE_THEORA $(shell pkg-config --cflags theoraenc)
VIDEO_LIBS = $(shell pkg-config --libs theoraenc)
endif

build/ray-tracing: main.cc trace_kernel.hh gpu_opencl.hh minicl.hpp random.hh vector.hh ray.hh color.hh scene.hh theora.hh $(VIDEO_SOURCE) Makefile
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_CXXFLAGS) $(VIDEO_FLAGS) main.cc $(VIDEO_SOURCE) -o $@ $(LDFLAGS) $(OPENMP_LIBS) $(OPENCL_LIB) $(VIDEO_LIBS) $(LDLIBS)

.PHONY: test benchmark clean
test: build/ray-tracing
	cd build && ./ray-tracing --test
benchmark: build/ray-tracing
	cd build && ./ray-tracing --benchmark
clean:
	rm -rf build

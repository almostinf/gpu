CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
# Prefer official pkg-config metadata; fallback to the ICD-loader SONAME on Linux.
# Override explicitly: make OPENCL_LIB=-lOpenCL
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

build/super-boring-task: main.cc linear-algebra.hh gpu_opencl.hh minicl.hpp Makefile
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_CXXFLAGS) main.cc -o $@ $(LDFLAGS) $(OPENCL_LIB) $(OPENMP_LIBS) $(LDLIBS)

.PHONY: clean
clean:
	rm -rf build

.PHONY: cpu-test
cpu-test:
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_CXXFLAGS) -I. tests/cpu_reference.cc -o build/cpu-test $(LDFLAGS) $(OPENMP_LIBS) $(LDLIBS)
	./build/cpu-test

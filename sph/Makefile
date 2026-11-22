CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
CXXFLAGS += -ffp-contract=off
ifeq ($(shell uname -s),Darwin)
LIBOMP_PREFIX ?= $(shell brew --prefix libomp)
OPENMP_CXXFLAGS ?= -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
OPENMP_LIBS ?= -L$(LIBOMP_PREFIX)/lib -Wl,-rpath,$(LIBOMP_PREFIX)/lib -lomp
OPENCL_LIB ?= -framework OpenCL
WITH_GUI ?= 1
GUI_FLAGS = -DGL_SILENCE_DEPRECATION -Wno-deprecated-declarations
GUI_LIBS = -framework OpenGL -framework GLUT
else
OPENMP_CXXFLAGS ?= -fopenmp
OPENMP_LIBS ?= -fopenmp
OPENCL_LIB ?= $(shell pkg-config --libs OpenCL 2>/dev/null || echo -l:libOpenCL.so.1)
WITH_GUI ?= $(shell pkg-config --exists glut gl 2>/dev/null && echo 1 || echo 0)
GUI_FLAGS = $(shell pkg-config --cflags glut gl 2>/dev/null)
GUI_LIBS = $(shell pkg-config --libs glut gl 2>/dev/null)
endif
ifeq ($(WITH_GUI),1)
DISPLAY_FLAGS = -DHAVE_GUI $(GUI_FLAGS)
DISPLAY_LIBS = $(GUI_LIBS)
WITH_THEORA ?= $(shell pkg-config --exists theoraenc 2>/dev/null && echo 1 || echo 0)
ifeq ($(WITH_THEORA),1)
VIDEO_SOURCE = theora.cc
VIDEO_FLAGS = -DHAVE_THEORA $(shell pkg-config --cflags theoraenc)
VIDEO_LIBS = $(shell pkg-config --libs theoraenc)
endif
endif

build/sph: main.cc sph.hh sph_kernel.hh gpu_sph.hh gpu_opencl.hh minicl.hpp opengl.hh vector.hh theora.hh $(VIDEO_SOURCE) Makefile
	@mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_CXXFLAGS) $(DISPLAY_FLAGS) $(VIDEO_FLAGS) main.cc $(VIDEO_SOURCE) -o $@ $(LDFLAGS) $(OPENMP_LIBS) $(OPENCL_LIB) $(DISPLAY_LIBS) $(VIDEO_LIBS) $(LDLIBS)

.PHONY: test benchmark run vglrun clean
test: build/sph
	cd build && ./sph --test
benchmark: build/sph
	cd build && ./sph --benchmark
run: build/sph
	./build/sph
vglrun: build/sph
	env LD_LIBRARY_PATH=$(LIBRARY_PATH) vglrun ./build/sph
clean:
	rm -rf build

CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -mbmi2 -mavx2
LDFLAGS =

# HDF5 flags
HDF5_CFLAGS = -I/usr/include/hdf5/serial
HDF5_LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5

# PyBind11 flags (use python3-config for includes)
PYTHON_CFLAGS = $(shell python3 -m pybind11 --includes) -fPIC
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)
PYTHON_EXT = $(shell python3-config --extension-suffix)

# Directories
SRCDIR = .
CORE = core
SEARCH = search
TABLEBASE = tablebase
NN = nn
BINDIR = bin
OBJDIR = obj

# Source files
CORE_SRCS = $(CORE)/board.cpp $(CORE)/movegen.cpp $(CORE)/notation.cpp
SEARCH_SRCS = $(SEARCH)/search.cpp $(SEARCH)/tt.cpp
TB_SRCS = $(TABLEBASE)/tablebase.cpp $(TABLEBASE)/compression.cpp
NN_SRCS = $(NN)/mlp.cpp

# Object files
CORE_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(CORE_SRCS))
SEARCH_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SEARCH_SRCS))
TB_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(TB_SRCS))
NN_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(NN_SRCS))

ALL_OBJS = $(CORE_OBJS) $(SEARCH_OBJS) $(TB_OBJS) $(NN_OBJS)

# Targets
all: dirs $(BINDIR)/test_search $(BINDIR)/perft $(BINDIR)/generate_training $(BINDIR)/test_nn $(BINDIR)/match $(BINDIR)/play $(BINDIR)/genbook $(BINDIR)/viewbook $(BINDIR)/condensebook

# Python module
python: dirs dtm_sampler$(PYTHON_EXT)

dirs:
	@mkdir -p $(BINDIR) $(OBJDIR)/$(CORE) $(OBJDIR)/$(SEARCH) $(OBJDIR)/$(TABLEBASE) $(OBJDIR)/$(NN)

$(BINDIR)/test_search: $(ALL_OBJS) $(OBJDIR)/test_search.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/perft: $(CORE_OBJS) $(OBJDIR)/perft.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/generate_training: $(ALL_OBJS) $(OBJDIR)/generate_training.o
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ $(HDF5_LDFLAGS)

$(BINDIR)/test_nn: $(CORE_OBJS) $(NN_OBJS) $(OBJDIR)/test_nn.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/match: $(ALL_OBJS) $(OBJDIR)/match.o
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^

$(BINDIR)/play: $(ALL_OBJS) $(OBJDIR)/play.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/genbook: $(ALL_OBJS) $(OBJDIR)/genbook.o
	$(CXX) $(LDFLAGS) -pthread -o $@ $^

$(BINDIR)/viewbook: $(CORE_OBJS) $(OBJDIR)/viewbook.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/condensebook: $(CORE_OBJS) $(OBJDIR)/condensebook.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Object file rules
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test executables
$(OBJDIR)/test_search.o: test_search.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/perft.o: perft.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/generate_training.o: generate_training.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $(HDF5_CFLAGS) -c $< -o $@

$(OBJDIR)/test_nn.o: test_nn.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/match.o: match.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

$(OBJDIR)/play.o: play.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/genbook.o: genbook.cpp
	$(CXX) $(CXXFLAGS) -pthread -c $< -o $@

$(OBJDIR)/viewbook.o: viewbook.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/condensebook.o: condensebook.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Python module: dtm_sampler (needs -fPIC objects)
PIC_OBJDIR = obj_pic

pic_dirs:
	@mkdir -p $(PIC_OBJDIR)/$(CORE) $(PIC_OBJDIR)/$(TABLEBASE)

$(PIC_OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

PIC_CORE_OBJS = $(patsubst %.cpp,$(PIC_OBJDIR)/%.o,$(CORE_SRCS))
PIC_TB_OBJS = $(patsubst %.cpp,$(PIC_OBJDIR)/%.o,$(TB_SRCS))

dtm_sampler$(PYTHON_EXT): pic_dirs $(PIC_OBJDIR)/dtm_sampler.o $(PIC_OBJDIR)/dtm_sampler_py.o $(PIC_TB_OBJS) $(PIC_CORE_OBJS)
	$(CXX) -shared -o $@ $(filter-out pic_dirs,$^) $(LDFLAGS)

$(PIC_OBJDIR)/dtm_sampler.o: dtm_sampler.cpp dtm_sampler.hpp
	$(CXX) $(CXXFLAGS) $(PYTHON_CFLAGS) -c $< -o $@

$(PIC_OBJDIR)/dtm_sampler_py.o: dtm_sampler_py.cpp dtm_sampler.hpp
	$(CXX) $(CXXFLAGS) $(PYTHON_CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(PIC_OBJDIR) $(BINDIR) dtm_sampler*.so

# WebAssembly build
wasm:
	$(MAKE) -C web -f Makefile.wasm

wasm-clean:
	$(MAKE) -C web -f Makefile.wasm clean

wasm-debug:
	$(MAKE) -C web -f Makefile.wasm debug

.PHONY: all dirs clean python pic_dirs wasm wasm-clean wasm-debug

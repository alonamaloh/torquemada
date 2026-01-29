CXX = g++
CXXFLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -mbmi2 -mavx2 -fopenmp
LDFLAGS = -fopenmp

# Directories
SRCDIR = .
CORE = core
SEARCH = search
TABLEBASE = tablebase
BINDIR = bin
OBJDIR = obj

# Source files
CORE_SRCS = $(CORE)/board.cpp $(CORE)/movegen.cpp $(CORE)/notation.cpp
SEARCH_SRCS = $(SEARCH)/search.cpp $(SEARCH)/tt.cpp
TB_SRCS = $(TABLEBASE)/tablebase.cpp $(TABLEBASE)/compression.cpp

# Object files
CORE_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(CORE_SRCS))
SEARCH_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SEARCH_SRCS))
TB_OBJS = $(patsubst %.cpp,$(OBJDIR)/%.o,$(TB_SRCS))

ALL_OBJS = $(CORE_OBJS) $(SEARCH_OBJS) $(TB_OBJS)

# Targets
all: dirs $(BINDIR)/test_search $(BINDIR)/perft $(BINDIR)/selfplay

dirs:
	@mkdir -p $(BINDIR) $(OBJDIR)/$(CORE) $(OBJDIR)/$(SEARCH) $(OBJDIR)/$(TABLEBASE)

$(BINDIR)/test_search: $(ALL_OBJS) $(OBJDIR)/test_search.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/perft: $(CORE_OBJS) $(OBJDIR)/perft.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(BINDIR)/selfplay: $(ALL_OBJS) $(OBJDIR)/selfplay.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Object file rules
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test executables
$(OBJDIR)/test_search.o: test_search.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/perft.o: perft.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/selfplay.o: selfplay.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all dirs clean

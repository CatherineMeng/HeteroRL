CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

BUFFER_EXE_NAME = rmm-buffers
BUFFER_SOURCES = src/host.cpp


all: build_buffers

build_buffers:
	$(CXX) $(CXXFLAGS) -o $(BUFFER_EXE_NAME) $(BUFFER_SOURCES)

run: 
	./$(BUFFER_EXE_NAME)

clean: 
	rm -f $(BUFFER_EXE_NAME) $(USM_EXE_NAME)

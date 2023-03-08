# Test Instructions for Replay Buffer Implementation using Intel DevCloud

## Uniform Replay Buffer

Create CMakeLists.txt with the following content:

```
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
endif()

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(uniform_replay CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(uniform_replay uniform_replay.cpp replay_buffer_base.cpp)
target_link_libraries(uniform_replay "${TORCH_LIBRARIES}")
set_property(TARGET uniform_replay PROPERTY CXX_STANDARD 17)
```

Build:
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/home/u186670/libtorch ..
$ cmake --build . --config Release
```

(Optional) Request a CPU/GPU node:
```
$ qsub -I -l nodes=1:xeon:ppn=2 -d .
or
$ qsub -I -l nodes=1:gpu:ppn=2 -d .
```

Run:
```
$ ./uniform_replay
```

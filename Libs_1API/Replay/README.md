# Test Instructions for Replay Buffer Implementation using Intel DevCloud

## Pre-requisite
libtorch
On Devcloud: install libtorch at user root directory https://pytorch.org/cppdocs/installing.html

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
Note: My uID on Devcloud is u186670. Replace the ID in the absolute path for linking libtorch with your own user ID.

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

## Prioritized Replay Buffer with DPC++

Create CMakeLists.txt with the following content:
```
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
endif()

set(TARGET_NAME prioritized_replay)
#set(SOURCE_FILE prioritized_replay.cpp)
set(SOURCE_FILE vadd.cpp)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(${TARGET_NAME} CXX)

find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ${TORCH_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")

#add_executable(${TARGET_NAME} prioritized_replay.cpp replay_buffer_base.cpp sum_tree_nary.cpp)
add_executable(${TARGET_NAME} ${SOURCE_FILE})
# new content
set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})
# end new content
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
```

## Example (sum_tree_nary) with DPC++ and Libtorch 
### build commands for version including libtorch:
Request a compute node:
```
qsub -I -l nodes=1:gpu:ppn=2 -d .
```

Set environment variable:
```
export LD_LIBRARY_PATH=/glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/lib/:$LD_LIBRARY_PATH
```

Compile:

```
icpx -fsycl sum_tree_nary.cpp -I /glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I /glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/include -L /glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -lgomp -w
```

Run:
```
./a.out
```
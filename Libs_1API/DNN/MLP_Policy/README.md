## Pre-requisite
libtorch
On Devcloud: install libtorch at user root directory https://pytorch.org/cppdocs/installing.html

## Actor - Policy Inference only:

Create CMakeLists.txt with the following content:

```
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
endif()

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(dqn CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(dqn dqn.cpp)
target_link_libraries(dqn "${TORCH_LIBRARIES}")
set_property(TARGET dqn PROPERTY CXX_STANDARD 17)
```

Build:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/u186670/libtorch ..
cmake --build . --config Release
```

Run random simulation test (DevCloud):
```
./dqn
```

## Learner (DevCloud):

Create CMakeLists.txt with the following content:

```
if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
endif()

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(learner CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(learner src/learner.cpp)
target_link_libraries(learner "${TORCH_LIBRARIES}")
set_property(TARGET learner PROPERTY CXX_STANDARD 17)
```

Build:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/u186670/libtorch ..
cmake --build . --config Release
```

Run random simulation test (DevCloud):
```
./learner
```

## Learner (Kalu):

in /home/yuan/libtorch/share/cmake/Caffe2/public/cuda.cmake
find the line 
```
  # Get cuDNN version
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
```
change it to
```
  # Get cuDNN version
  file(READ ${CUDNN_INCLUDE_DIR}/cudnn_version.h CUDNN_HEADER_CONTENTS)
```

Create CMakeLists.txt with the following content:
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(learner)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(learner src/learner_kalu.cpp)
target_link_libraries(learner "${TORCH_LIBRARIES}")
set_property(TARGET learner PROPERTY CXX_STANDARD 17)

set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so.8.9.4)
set(CUDNN_LIBRARY_PATH /usr/local/cuda/lib64)
```

build:
```
cmake -DCMAKE_PREFIX_PATH=/home/yuan/libtorch ..
cmake --build . --config Release
or
make
```
Run random simulation test (Kalu):
```
./learner
```

## Learner - Python
```
python mlp_dqn.py
python mlp_ddpg.py
```
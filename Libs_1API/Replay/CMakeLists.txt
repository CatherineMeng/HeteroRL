if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
endif()

set(TARGET_NAME prioritized_replay)
set(SOURCE_FILE prioritized_replay.cpp)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(${TARGET_NAME} CXX)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl ${TORCH_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lOpenCL -lsycl")

add_executable(${TARGET_NAME} ${SOURCE_FILE})

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

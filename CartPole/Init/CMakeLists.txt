cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(CartPole)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(CartPole CartPole.cpp)
target_link_libraries(CartPole "${TORCH_LIBRARIES}")
set_property(TARGET CartPole PROPERTY CXX_STANDARD 14)

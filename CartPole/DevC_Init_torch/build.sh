mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/u186670/libtorch ..
cmake --build . --config Release

g++ -o notorchout CP_notorch.cpp
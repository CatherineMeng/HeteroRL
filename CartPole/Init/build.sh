mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/yuan/libtorch ..
cmake --build . --config Release

g++ -o notorchout CP_notorch.cpp
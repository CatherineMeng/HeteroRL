### (DevCloud only) Access GPU node	
```
$qsub -I -l nodes=1:gpu:ppn=2 -d .
or
$qsub -l nodes=1:gpu:ppn=2 -d .
or,
list all gen9 gpu nodes
$pbsnodes | grep -B4 gpu
select one <node id> with state = free
$qsub -I -l nodes=1:<node id>:ppn=2
```
Monitor GPU usage:
```
intel_gpu_top
```

### Compile
```
icpx -std=c++17 -fsycl -g -o trainer_dpc MLP_train_sycl.cpp 
```
### Run
```
./trainer_dpc
```

Note: devcloud, compile and run mm sample:
```
$qsub -I -l nodes=1:<node id>:ppn=2
logged into s019-n020
$ lscpu
    11th Gen Intel(R) Core(TM) i9-11900KB @ 3.30GHz, 16 cores
intel_gpu_top not installed
GPU (according to https://www.notebookcheck.net/Intel-Core-i9-11900KB-Processor-Benchmarks-and-Specs.588996.0.html): Intel UHD Graphics Xe 32EUs (Tiger Lake-H) (350 - 1450 MHz), 32 - unified pipelines
$ cd /home/u186670/oneAPI-samples/DirectProgramming/C++SYCL/DenseLinearAlgebra/matrix_mul
$ icpx -std=c++17 -fsycl -g -o matrix_mul_dpc src/matrix_mul_sycl.cpp 
$ ./matrix_mul_dpc 
output:
    Device: Intel(R) UHD Graphics [0x9a60]
    Problem size: c(150,600) = a(150,300) * b(300,600)
    init A submitted;
    init B submitted;
    Result of matrix multiplication using SYCL: Success - The results are correct! 
```

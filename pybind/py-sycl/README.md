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

### Compile (C++)
```
icpx -std=c++17 -fsycl -g -o trainer_dpc MLP_train_sycl.cpp 
```
### Run (C++)
```
./trainer_dpc
```


## Testing Pybind
### Devcloud: Installation
```
conda activate base
pip install pybind11
```
### Compile (PyBind)
Build binding library:
```
icpx -shared -std=c++17 -fsycl -fPIC $(python3 -m pybind11 --includes) -I /home/u186670/.local/lib/python3.9/site-packages/pybind11/include bindings_learner.cpp -o sycl_learner_module$(python3-config --extension-suffix --includes)
```
### Run (Python)
```
python test_binded_lib.py
```
Tested on Aug16 Devcloud, Run (Python) results same as Run (C++)
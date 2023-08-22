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

## Learner - SYCL
### Compile (C++)
```
icpx -std=c++17 -fsycl -g -o trainer_dpc MLP_train_sycl.cpp 
```
### Run (C++)
```
./trainer_dpc
```

## Learner - Testing Pybind
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


## Prioritized Replay Manager - SYCL
### Compile (C++)
```
icpx -fsycl -lgomp sum_tree_nary.cpp -o out
```
### Run (C++)
```
./out
```

## Prioritized Replay Manager - Testing Pybind
### Compile(Pybind)
Build binding library:
```
$conda activate htroRL
$icpx -shared -std=c++17 -fsycl -fPIC $(python3 -m pybind11 --includes) -I /home/yuan/.conda/envs/htroRL/lib/python3.8/site-packages/pybind11/include bindings_replay.cpp -o sycl_rm_module$(python3-config --extension-suffix --includes)
```
### Run (Python)
```
python test_binded_lib_rm.py
```
Tested on Aug21 on Kalu, Run (Python) results same as Run (C++)

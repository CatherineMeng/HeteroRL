## CPU-GPU
### Compile:
```
icpx -fsycl sum_tree_nary.cpp -o out
```

### Run-CPU:
```
qsub -I -l nodes=1:xeon:ppn=2 -d .
./out
```
### Run-GPU:
```
qsub -l nodes=1:gpu:ppn=2 -d .
./out
```
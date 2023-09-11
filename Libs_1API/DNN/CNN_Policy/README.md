
### devcloud
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
icpx -fsycl src/learner.cpp -I /glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I /glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/include -L /glob/development-tools/versions/oneapi/2023.0.1/oneapi/intelpython/latest/envs/pytorch/lib/python3.9/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -w
```

Run:
```
./a.out
```


### Kalu

Run:
```
conda activate htroRLatari
python cnn.py
```
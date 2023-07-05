## MM / MLP inference / MLP training pipeline on FPGA - Devcloud

Note: before compiling, copy the \include folder from ../Replay-FPGA-autorun into src.
```
mkdir src/include
cp -r ../Replay-FPGA-autorun/src/include/. src/include
```

### FPGA Emulation

DevCloud, June 4

```
$ qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
mkdir build
cd build
cmake ..
```
software emulation: 
```
make fpga_emu
```

### FPGA Report

```
make report
```
View report: download build/rmmfpga_report.prj
Open build/mvfpga_report.prj/reports/report.html        

### FPGA Hardware Simulation

Re-organized with src folder and cmake:
```
make fpga_sim
```
### FPGA Hardware

Re-organized with src folder and cmake:
```
make fpga
icpx -fsycl -fintelfpga ../src/host.cpp -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10 -o link.fpga
```

icpx -fsycl -fintelfpga ../src/simple_host_streaming.cpp -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10 -o link.fpga

Available Nodes	Command Options
FPGA Compile Time	
```
qsub -I -l nodes=1:fpga_compile:ppn=2 -d .
```

FPGA Runtime	
```
qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
```

GPU	
```
qsub -I -l nodes=1:gpu:ppn=2 -d .
```

CPU	
```
qsub -I -l nodes=1:xeon:ppn=2 -d .
```


## MM / MLP inference / MLP training pipeline on FPGA - Local machine DE10
### FPGA Emulation
July 4
Using host_buffer.cpp
If on Kalu:
```
source /opt/xilinx/xrt/setup.sh
```
Compile:
```
make -f Makefile.fpga fpga_emu
```
Run:
```
./mlp-buffers.fpga_emu
```
Clean up:
```
make -f Makefile.fpga fpga_emu
```

## Testing Pybind
Build binding library (example):
```
$ conda activate htroRL
$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
<!-- This should output a file: example.cpython-38-x86_64-linux-gnu-->
```
Test binding library (example - add):
```
$ python
>>> import example
>>> example.add(1, 2)
```
Test binding library (example - Pet):
```
$ python
>>> from example import *
>>> pet1 = Pet("dave")
>>> pet1.setName("a")
>>> pet1.getName() #return 'a'
```
from example import *

Build binding library for replay (software emulation):
```
$ conda activate htroRL
$ dpcpp -O3 -Wall -shared -std=c++17 -fintelfpga -fPIC $(python3 -m pybind11 --includes) -I /home/yuan/.conda/envs/htroRL/lib/python3.8/site-packages/pybind11/include bindings.cpp -o replay_module$(python3-config --extension-suffix --includes) -DFPGA_EMULATOR=1
<!-- This should output a file: replay.cpython-38-x86_64-linux-gnu-->
```

Build binding library for replay (hardware):
```
$ conda activate htroRL
$ 
<!-- This should output a file: replay_hw.cpython-38-x86_64-linux-gnu-->
```

## Testing FPGA emu
Build replay class in dpcpp:
```
make -f Makefile.fpga fpga_emu
./rmm-buffers.fpga_emu
make -f Makefile.fpga clean
```

test replay class in dpcpp:
```
nohup make -f Makefile.fpga hw > mylog.txt & disown
./rmm-buffers-hwtest.fpga
```

### debugging inconsistent dpcpp code behavior between fpga_emu executable and pybind library
python compile:
```
dpcpp -O3 -Wall -shared -std=c++17 -fintelfpga -fPIC $(python3 -m pybind11 --includes) -I /home/yuan/.conda/envs/htroRL/lib/python3.8/site-packages/pybind11/include bindings.cpp -o replay_module$(python3-config --extension-suffix --includes) -DFPGA_EMULATOR=1
```

dpcpp compile:
```
dpcpp -O2 -g -std=c++17 -fintelfpga -fPIC $(python3 -m pybind11 --includes) host_dpcpp.cpp -o rmm-buffers.fpga_emu(python3-config --extension-suffix) -DFPGA_EMULATOR=1
```
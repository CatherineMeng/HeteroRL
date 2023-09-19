
## FPGA sw
```
make -f Makefile.fpga fpga_emu
./example.fpga
```

## FPGA hw

```
nohup make -f Makefile.fpga hw > mylog.txt & disown
./example.fpga
```

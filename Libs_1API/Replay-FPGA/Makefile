CXX = icpx
CXXFLAGS =  -g -fsycl -std=c++17 -I.

# target:dependency
# $@ - target; $^ - all dependencies

# Apr 18 tried on devcloud, succeed
# requested $ qsub -I -l nodes=1:fpga_compile:ppn=2 -d .
fpga_emu: kernel.cpp  host.cpp
	$(CXX) $(CXXFLAGS) -fintelfpga -DFPGA_EMULATOR $^ -o $@	

# report - default fpga device:
# Apr 18 tried on devcloud, succeed. 
# requested $ qsub -I -l nodes=1:fpga_compile:ppn=2 -d .
# report viewed by downloading fpga_compile_report.prj locally and opening reports/report.html
default_fpga_report: kernel.cpp
	$(CXX) $(CXXFLAGS) -fintelfpga -Xshardware -fsycl-link=early kernel.cpp -o fpga_compile_report.a

# report - explicit fpga device:
fpga.report: hardware.o
	$(CXX) $(CXXFLAGS) -fintelfpga -fsycl-link $^ -o $@ -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10


# FPGA hardware (default FPGA device)
# Apr 18 tried on devcloud, succeed
# requested qsub -I -l nodes=1:fpga_runtime:ppn=2 -d .
# used command: # icpx -fsycl -fintelfpga host.cpp kernel.cpp -Xshardware -o link.fpga
default_fpga_hardware: kernel.cpp
	$(CXX) $(CXXFLAGS) -fintelfpga -Xshardware kernel.cpp -o fpga_compile.fpga
	

# FPGA hardware (explicit FPGA device)
fpga_hardware: kernel.cpp
	icpx -fintelfpga -Xshardware -Xstarget=intel_s10sx_pac:pac_s10 kernel.cpp -o fpga_compile.fpga



fpga_hardware.fpga: hardware.o
	$(CXX) $(CXXFLAGS) -fintelfpga $^  -o $@  -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10

hardware.o: kernel.cpp
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1


dev.o: kernel.cpp
	$(CXX) -fintelfpga -c -o dev.o kernel.cpp

dev_image.a: dev.o
	$(CXX) -fintelfpga -fsycl-link=image dev.o -o dev_image.a -Xshardware -Xsboard=/opt/intel/oneapi/intel_s10sx_pac:pac_s10


host.o: host.cpp
	$(CXX) -fintelfpga -c -o host.o host.cpp 

dev_link.fpga: dev_image.a host.o
	$(CXX) -fintelfpga host.o dev_image.a -o dev_link.fpga


clean:
	rm *.o *.fpga *.a
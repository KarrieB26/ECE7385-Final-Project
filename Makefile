NVCC   = nvcc
ARCH   = -arch=sm_80
CFLAGS = -O3 -Iinclude/
BINDIR = bin

all: hist_bench tri_test

$(BINDIR):
	mkdir -p $(BINDIR)

hist_bench: src/main.cu src/cpu_baseline.cu include/utils.cuh include/histogram.cuh include/cpu_baseline.cuh | $(BINDIR)
	$(NVCC) $(ARCH) $(CFLAGS) src/main.cu src/cpu_baseline.cu -o $(BINDIR)/hist_bench

tri_test: src/scan_test.cu include/utils.cuh include/scan.cuh | $(BINDIR)
	$(NVCC) $(ARCH) $(CFLAGS) src/scan_test.cu -o $(BINDIR)/tri_test

clean:
	rm -rf $(BINDIR) *.log logs/*.log
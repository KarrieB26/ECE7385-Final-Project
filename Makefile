NVCC   = nvcc
ARCH   = -arch=sm_80
CFLAGS = -O3 -Iinclude/
TARGET = hist_bench

all: $(TARGET)

$(TARGET): src/main.cu src/cpu_baseline.cu include/utils.cuh include/histogram.cuh include/cpu_baseline.cuh
	$(NVCC) $(ARCH) $(CFLAGS) src/main.cu src/cpu_baseline.cu -o $(TARGET)

clean:
	rm -f $(TARGET) *.log
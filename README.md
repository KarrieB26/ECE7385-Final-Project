# **GPU Accelerated Histogram and Prefix Sum Optimization**

This project focuses on implementing and optimizing high-performance GPU kernels for **Histogram Computation** and **Prefix Sum (Scan)**. It compares a single-threaded C++ baseline against multiple CUDA implementations to solve the "atomic bottleneck" and maximize memory throughput on NVIDIA A100/H100 architectures.

## **Project Structure**

The project follows a modular separate-compilation architecture to maintain clean separation between host orchestration and device-side kernels.

```text
project/
├── include/
│   ├── utils.cuh            # CUDA error checking (cudaCheckError) & timing helpers
│   ├── histogram.cuh        # __global__ kernel definitions (Naive, Shared, Privatized)
│   ├── scan.cuh             # __global__ scan kernel definitions (Kogge-Stone, Brent-Kung)
│   └── stb_image.h          # Single-header image loading library
├── src/
│   ├── main.cu              # Host orchestration, memory management, and kernel launches
│   ├── data_loader.cu       # Image ingestion and flattening logic
│   └── cpu_baseline.cu      # Sequential C++ ground truth for validation
├── data/
│   └── chest_xray/          # Kaggle Chest X-Ray (Pneumonia) dataset (JPEGs)
├── scripts/
│   ├── run_bench.sh         # Slurm sbatch script for production benchmarks
│   └── run_profile.sh       # Nsight Compute profiling script
└── Makefile                 # Build system targeting sm_80 (A100) or sm_90 (H100)
```

## **Team Roles & Deliverables**

* **Sneha (Data Lead):** Dataset ingestion, JPEG flattening via `stb_image.h`, and host-to-device transfer logic.
* **Karrie (Algorithm Lead):** Sequential C++ baseline and the `validate()` function for CDF/Histogram accuracy.
* **Tri (Infrastructure Lead):** CUDA environment setup, kernel implementation, and performance profiling via Nsight Compute.

## **Getting Started**

### **Prerequisites**
* **Hardware:** Access to NVIDIA A100 or H100 GPU nodes on the SMU SuperPod.
* **Environment:** CUDA Toolkit 12.x and `nvcc` compiler.
* **Dataset:** Download the [Chest X-Ray (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) into the `data/` directory.

### **Build Instructions**
Use the provided `Makefile` to compile the project. The default architecture is set to `sm_80` (A100).
```bash
make clean
make
```

### **Running the Baseline**
To run the project in an interactive session:
```bash
srun --gres=gpu:1 --pty bash
./hist_bench
```

### **Profiling with Nsight Compute**
To capture hardware metrics (Warp Occupancy, DRAM Throughput, Atomic Contention):
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum.per_second,\
smsp__sass_inst_executed_op_atom.sum \
./hist_bench
```

## **Performance Goals**
* **Target Speedup:** Achieve a $10x–50x$ improvement over the CPU baseline for datasets $> 10^7$ elements.
* **Occupancy:** Maintain $> 75\%$ Warp Occupancy on A100 hardware.
* **Validation:** All GPU outputs must pass the `validate()` check against the CPU ground truth.

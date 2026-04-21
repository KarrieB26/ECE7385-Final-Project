#!/bin/bash
#SBATCH --job-name=ECE7385_Final_Integration
#SBATCH --output=logs/final_run_%j.log
#SBATCH --error=logs/final_run_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --account=camp_ece5385_1262_000_0001

# Modules (Using your working Lab 1 setup)
module purge
module load gcc
module load cuda

# Compile right before running to ensure everything is fresh
echo "Compiling ECE 7385 Project..."
make clean
make
echo ""

# Run the multi-image benchmark
echo "Starting Integration Handshake (X-Ray Data)..."
./hist_bench
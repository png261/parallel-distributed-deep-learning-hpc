#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/cuda-12.1/bin:/opt/anaconda3/envs/pylab/bin:/opt/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p build results/logs results/figures

echo "=========================================================="
echo "[Step 1/4] Compiling CUDA C++ Benchmarks (sm_61: GTX 1080 Ti & GTX 1080)"
echo "=========================================================="
nvcc -O3 -arch=sm_61 -Xcompiler -fopenmp src/cuda/conv_fc_benchmark.cu -o build/conv_fc_benchmark -lcublas
nvcc -O3 -arch=sm_61 -Xcompiler "-fPIC -fopenmp" -shared src/cuda/libgpu_cnn.cu -o build/libgpu_cnn.so
./build/conv_fc_benchmark | tee results/logs/cuda_benchmark.log

echo ""
echo "=========================================================="
echo "[Step 2/4] Running Section 2 Multi-Core CPU (1..16 cores) & Multi-GPU Benchmark"
echo "=========================================================="
/opt/anaconda3/envs/pylab/bin/python src/distributed/multicore_multigpu_benchmark.py | tee results/logs/multicore_multigpu.log

echo ""
echo "=========================================================="
echo "[Step 3/4] Running Section 4 & 5 Distributed Deep Learning Algorithms"
echo "           (ShuffleSGD, ParallelSGD, ADMM.SGD, DownpourSGD)"
echo "=========================================================="
/opt/anaconda3/envs/pylab/bin/python src/distributed/distributed_sgd_algorithms.py | tee results/logs/distributed_sgd.log

echo ""
echo "=========================================================="
echo "[Step 4/4] Generating Publication Figures (Figures 1 to 7)"
echo "=========================================================="
/opt/anaconda3/envs/pylab/bin/python src/plot_results.py

echo ""
echo "All experiments completed successfully!"

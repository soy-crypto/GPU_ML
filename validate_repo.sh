#!/bin/bash

set -e

# Find repo root (directory of this script)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===================================="
echo "GPU Foundations Repository Validator"
echo "Repo root: $ROOT"
echo "===================================="

echo ""
echo "1️⃣ Checking C++ inference runtime build..."

cd "$ROOT/cpp_inference_runtime"

rm -rf build
mkdir build
cd build

cmake .. > /dev/null
make

echo "C++ inference runtime build: OK"

echo ""
echo "2️⃣ Checking CUDA kernel compilation..."

cd "$ROOT/cuda_kernel_optimization"

echo "Compiling vector_add..."
nvcc -c vector_add/main.cu -o /tmp/vector_add.o

echo "Compiling naive GEMM..."
nvcc -c gemm_naive/main.cu -o /tmp/gemm_naive.o

echo "Compiling tiled GEMM..."
nvcc -c gemm_tiled/main.cu -o /tmp/gemm_tiled.o

echo "CUDA kernels compile: OK"

echo ""
echo "3️⃣ Checking CUDA microbenchmarks..."

cd "$ROOT/cuda_microbenchmarks/bandwidth_test"
make

echo "CUDA microbenchmarks build: OK"

echo ""
echo "4️⃣ Checking Python syntax..."

cd "$ROOT"

python3 -m py_compile llm_inference_systems/inference_benchmarks/*.py

echo "Python syntax check: OK"

echo ""
echo "===================================="
echo "All checks completed successfully"
echo "===================================="
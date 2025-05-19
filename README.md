# CUDA Tutorials

Welcome to the **CUDA Tutorials** repository! 🎮🚀

I am learning CUDA and CUDA is fun!

## 📁 Contents

- `hello.cu` – Simple example demonstrating `printf` from the GPU.
- `vectoradd.cu` – Basic vector addition using CUDA.
- `matrixadd.cu` – Basic matrix addition using CUDA.

## 🔧 Prerequisites

Before you begin, make sure you have:

- An NVIDIA GPU with CUDA support
- CUDA Toolkit installed (e.g., via [NVIDIA's website](https://developer.nvidia.com/cuda-downloads))

Check your `nvcc` installation:

```bash
nvcc --version
```

## 🚀 Getting Started
1. Clone the repository
```bash
git clone https://github.com/Malisha4065/CUDA-Tutorials
cd cuda-tutorials
```

2. Build and Run
```bash
nvcc -o vec_add vec_add.cu
./vec_add
```

Happy GPU programming! ⚡

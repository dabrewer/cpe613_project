#!/bin/bash
OUT_PATH=output
BIN_PATH=bin

# Ensure the output directories exist
mkdir -p $OUT_PATH
mkdir -p $OUT_PATH/post
mkdir -p $BIN_PATH

# Build cpu executables
g++ -Isrc/include/ src/main.cpp src/solve_cpu.cpp src/voxel.cpp -o $BIN_PATH/main_cpu -O3

# Build gpu executables
#g++ -Isrc/include/ src/main.cpp src/solve_gpu.cpp src/voxel.cpp -o $BIN_PATH/main_gpu -O3
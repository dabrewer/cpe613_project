#!/bin/bash
OUT_PATH=output
BIN_PATH=bin

# Ensure the output directories exist
mkdir -p $OUT_PATH
mkdir -p $OUT_PATH/post
mkdir -p $BIN_PATH

# Build executables
g++ -Isrc_serial/include/ src_serial/main.cpp src_serial/mesh.cpp src_serial/voxel.cpp -o $BIN_PATH/main_serial -O3
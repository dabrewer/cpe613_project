#!/bin/bash

####################################################################################################
# DEFINITIONS
####################################################################################################
# Path definitions
OUT_PATH=output
BIN_PATH=bin

# Run characteristics
CPU_SIZE_RANGE="10 15 20 25 30 35 40"
GPU_SIZE_RANGE="10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"

GRID_DIMS=""
BLOCK_DIMS=""

####################################################################################################
# PROGRAM EXECUTION FUNCTIONS
####################################################################################################
# Function to run the serial configuration 10 times for each size configuration
run_cpu()
{
    for i in {1..10}
    do
        for s in $CPU_SIZE_RANGE
        do
            ./bin/main_cpu $OUT_PATH/cpu_v_${s} $OUT_PATH/cpu_e_${s} $OUT_PATH/cpu_s_${s}
        done
    done 
}

# Function to run each CUDA configuration 10 times with default block/grid
run_gpu_arch()
{
    for i in {1..10}
    do
        for s in $GPU_SIZE_RANGE
        do
            ./bin/main_gpu $OUT_PATH/gpu_p_v_${s} $OUT_PATH/gpu_p_e_${s} $OUT_PATH/gpu_p_s_${s}
            ./bin/main_gpu $OUT_PATH/gpu_t_v_${s} $OUT_PATH/gpu_t_e_${s} $OUT_PATH/gpu_t_s_${s}
            ./bin/main_gpu $OUT_PATH/gpu_a_v_${s} $OUT_PATH/gpu_a_e_${s} $OUT_PATH/gpu_a_s_${s}
        done
    done 
}

# Function to run each block/grid dim value 10 times on Ampere
run_gpu_dim()
{
    for i in {1..10}
    do
        # for s in $GPU_SIZE_RANGE
        # do
        #     ./bin/main_gpu $OUT_PATH/gpu_a_v_${s} $OUT_PATH/gpu_a_e_${s} $OUT_PATH/gpu_a_s_${s}
        # done
    done 
}

####################################################################################################
# MAIN ROUTINE
####################################################################################################
run_cpu()
#run_gpu_arch()
#run_gpu_dim()
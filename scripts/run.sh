#!/bin/bash

####################################################################################################
# EXECUTE CPU SIZE RUNS
####################################################################################################
./run_cpu.sh

####################################################################################################
# EXECUTE GPU SIZE RUNS
####################################################################################################
GPU_TYPE=pascal
run_gpu run_gpu.sh < input_pascal
GPU_TYPE=volta
run_gpu run_gpu.sh < input_volta
GPU_TYPE=ampere
run_gpu run_gpu.sh < input_ampere

####################################################################################################
# EXECUTE GPU DIM RUNS
####################################################################################################
run_gpu run_gpu_dim.sh < input_ampere

####################################################################################################
# VERIFY CPU/GPU CORRECTNESS
####################################################################################################
# Compare Voltage CPU Checksum to Comparable GPU (Pascal, Volta, Ampere) Checksums 
for cpu_name in output/cpu/*.vsum
do
    # Compare voltage checksums
    for gpu_name in output/gpu/*.vsum
    do
        CMD=diff -qs cpu_name gpu_name
        echo $CMD >> output/compare.txt
        echo $($CMD) >> output/compare.txt
    done
done
# Compare Field CPU Checksum to Comparable GPU (Pascal, Volta, Ampere) Checksums 
for cpu_name in output/cpu/*.esum
do
    # Compare voltage checksums
    for gpu_name in output/gpu/*.esum
    do
        CMD=diff -qs cpu_name gpu_name
        echo $CMD >> output/compare.txt
        echo $($CMD) >> output/compare.txt
    done
done
#find . -type f -name "test.txt"

####################################################################################################
# POST-PROCESS DATA
####################################################################################################
# Extract minimum execution time from each configuration and visualize results
#TODO: Python
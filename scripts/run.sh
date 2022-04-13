#!/bin/bash

SCRIPT_PATH=scripts

####################################################################################################
# EXECUTE CPU SIZE RUNS
####################################################################################################
echo RUNNING CPU...
./$SCRIPT_PATH/run_cpu.sh

####################################################################################################
# EXECUTE GPU SIZE RUNS
####################################################################################################
echo RUNNING PASCAL...
GPU_TYPE=pascal
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_pascal
echo RUNNING VOLTA...
GPU_TYPE=volta
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_volta
echo RUNNING AMPERE...
GPU_TYPE=ampere
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_ampere

####################################################################################################
# EXECUTE GPU DIM RUNS
####################################################################################################
echo RUNNING DIM...
run_gpu $SCRIPT_PATH/run_gpu_dim.sh < $SCRIPT_PATH/input_ampere

####################################################################################################
# VERIFY CPU/GPU CORRECTNESS
####################################################################################################
# Compare Voltage CPU Checksum to Comparable GPU (Pascal, Volta, Ampere) Checksums
echo COMPARING VSUMS 
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
echo COMPARING ESUMS
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
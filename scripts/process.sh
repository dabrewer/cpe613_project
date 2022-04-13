#!/bin/bash

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

####################################################################################################
# POST-PROCESS DATA
####################################################################################################
# Extract minimum execution time from each configuration and visualize results
#TODO: Python
#!/bin/bash

####################################################################################################
# DEFINITIONS
####################################################################################################
OUT_PATH=output/gpu/$GPU_TYPE/
BIN_PATH=bin

if [ -z ${SAVE_OUTPUT} ]
then
    SAVE_OUTPUT=n
fi

if [ -z ${ITER} ]
then
    ITER=1
fi

####################################################################################################
# Run characteristics
####################################################################################################
GPU_SIZE_RANGE="8 16 24 32 40 64 96 128 256 512 1024"

####################################################################################################
# PROGRAM EXECUTION FUNCTIONS
####################################################################################################
for i in {1..$ITER}
do
    for s in $GPU_SIZE_RANGE
    do
        ./bin/main_gpu ${s} 8 8 8 $OUT_PATH/${s}.v $OUT_PATH/${s}.e $OUT_PATH/${s}.s $SAVE_OUTPUT >> $OUT_PATH/gpu_${i}_${s}.out
    done
done

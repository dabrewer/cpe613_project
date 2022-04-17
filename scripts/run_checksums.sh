#!/bin/bash

SCRIPT_PATH=scripts

export ITER=1
export SAVE_OUTPUT=y

####################################################################################################
# PROGRAM EXECUTION FUNCTIONS
####################################################################################################
echo RUNNING CPU...
./$SCRIPT_PATH/run_cpu.sh

echo RUNNING PASCAL...
export GPU_TYPE=pascal
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_pascal

echo RUNNING VOLTA...
export GPU_TYPE=volta
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_volta

echo RUNNING AMPERE...
export GPU_TYPE=ampere
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_ampere


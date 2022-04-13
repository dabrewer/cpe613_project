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
export GPU_TYPE=pascal
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_pascal
echo RUNNING VOLTA...
export GPU_TYPE=volta
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_volta
echo RUNNING AMPERE...
export GPU_TYPE=ampere
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_ampere

####################################################################################################
# EXECUTE GPU DIM RUNS
####################################################################################################
echo RUNNING DIM...
run_gpu $SCRIPT_PATH/run_gpu_dim.sh < $SCRIPT_PATH/input_ampere
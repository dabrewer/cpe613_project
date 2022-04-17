#!/bin/bash

SCRIPT_PATH=scripts

export ITER=1
export SAVE_OUTPUT=y

####################################################################################################
# PROGRAM EXECUTION FUNCTIONS
####################################################################################################
echo RUNNING CPU...
./$SCRIPT_PATH/run_cpu.sh
shasum $(ls output/cpu/*.v) >> output/cpu/cpu.vsum
shasum $(ls output/cpu/*.e) >> output/cpu/cpu.esum

echo RUNNING PASCAL...
export GPU_TYPE=pascal
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_pascal
shasum $(ls output/gpu/pascal/*.v) >> output/gpu/pascal/pascal.vsum
shasum $(ls output/gpu/pascal/*.e) >> output/gpu/pascal/pascal.esum

echo RUNNING VOLTA...
export GPU_TYPE=volta
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_volta
shasum $(ls output/gpu/volta/*.v) >> output/gpu/volta/volta.vsum
shasum $(ls output/gpu/volta/*.e) >> output/gpu/volta/volta.esum

echo RUNNING AMPERE...
export GPU_TYPE=ampere
run_gpu $SCRIPT_PATH/run_gpu.sh < $SCRIPT_PATH/input_ampere
shasum $(ls output/gpu/ampere/*.v) >> output/gpu/ampere/ampere.vsum
shasum $(ls output/gpu/ampere/*.e) >> output/gpu/ampere/ampere.esum
#!/bin/bash

####################################################################################################
# DEFINITIONS
####################################################################################################
OUT_PATH=output/dim
BIN_PATH=bin

####################################################################################################
# Run characteristics
####################################################################################################
TILE_WIDTH_X=(1 2 4 8 16 32 64 128 256 512)
TILE_WIDTH_Y=(1 2 4 8 16 32 64 128 256 512)
TILE_WIDTH_Z=(1 2 4 8 16 32 64)

SIZE=512

####################################################################################################
# PROGRAM EXECUTION FUNCTIONS
####################################################################################################
for c in {1..10}
do
    for i in ${TILE_WIDTH_X[@]}
    do
        for j in ${TILE_WIDTH_Y[@]}
        do
            for k in ${TILE_WIDTH_Z[@]}
            do
                if [ $(($i * $j * $k)) -le 1024 ]
                then
                    #echo "$i" "$j" "$k";
                    ./bin/main_gpu "$SIZE" "$i" "$j" "$k" $OUT_PATH/$SIZE.v $OUT_PATH/$SIZE.e $OUT_PATH/$SIZE.s n >> $OUT_PATH/gpu_${c}_${i}_${j}_${k}.out
                fi
            done
        done
    done
done

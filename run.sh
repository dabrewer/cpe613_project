#!/bin/bash
module load cuda
./src/main_gpu 10 10 10 10 output/v.out output/f.out output/s.out

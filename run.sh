#!/bin/bash
module load cuda
./src/main_gpu 40 output/v.out output/f.out output/s.out >> output/out.txt

#!/bin/bash
module load cuda
./bin/main_gpu 1024 8 8 8 output/v.out output/f.out output/s.out >> output/out2.txt

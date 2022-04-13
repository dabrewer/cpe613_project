#!/bin/bash
module load cuda
./bin/main_gpu 500 10 10 10 output/v.out output/f.out output/s.out >> output/out2.txt

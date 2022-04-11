#!/bin/bash
module load cuda
<<<<<<< HEAD
./src/main_gpu 10 10 10 10 output/v.out output/f.out output/s.out
||||||| merged common ancestors
./src/main_gpu 10 output/v.out output/f.out output/s.out
=======
./src/main_gpu 40 output/v.out output/f.out output/s.out >> output/out.txt
>>>>>>> 35e0abf6e27a2cc9f4b665dbdcbb13fdacaf6993

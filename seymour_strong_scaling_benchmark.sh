#!/bin/bash

NOW=$(date "+%Y-%m-%dT%H%M%S")
mkdir -p output_seymour/$NOW

# one node: 1-8 GPUs
mpiexec -n 1 python mpiTesting.py -nx 8192 -ny 8192 --profile && 
mkdir -p output_seymour/$NOW/1_proc && 
mv *.log output_seymour/$NOW/1_proc/ && mv *.nc output_seymour/$NOW/1_proc/ &&

mpiexec -n 2 python mpiTesting.py -nx 8192 -ny 4096 --profile && 
mkdir -p output_seymour/$NOW/2_proc && 
mv *.log output_seymour/$NOW/2_proc/ && mv *.nc output_seymour/$NOW/2_proc/ &&

mpiexec -n 3 python mpiTesting.py -nx 8192 -ny 2731 --profile && 
mkdir -p output_seymour/$NOW/3_proc && 
mv *.log output_seymour/$NOW/3_proc/ && mv *.nc output_seymour/$NOW/3_proc/ &&

mpiexec -n 4 python mpiTesting.py -nx 8192 -ny 2048 --profile && 
mkdir -p output_seymour/$NOW/4_proc && 
mv *.log output_seymour/$NOW/4_proc/ && mv *.nc output_seymour/$NOW/4_proc/ &&

mpiexec -n 5 python mpiTesting.py -nx 8192 -ny 1638 --profile &&
mkdir -p output_seymour/$NOW/5_proc && 
mv *.log output_seymour/$NOW/5_proc/ && mv *.nc output_seymour/$NOW/5_proc/ &&

mpiexec -n 6 python mpiTesting.py -nx 8192 -ny 1365 --profile &&
mkdir -p output_seymour/$NOW/6_proc && 
mv *.log output_seymour/$NOW/6_proc/ && mv *.nc output_seymour/$NOW/6_proc/ &&

mpiexec -n 7 python mpiTesting.py -nx 8192 -ny 1170 --profile &&
mkdir -p output_seymour/$NOW/7_proc && 
mv *.log output_seymour/$NOW/7_proc/ && mv *.nc output_seymour/$NOW/7_proc/ &&

mpiexec -n 8 python mpiTesting.py -nx 8192 -ny 1024 --profile &&
mkdir -p output_seymour/$NOW/8_proc && 
mv *.log output_seymour/$NOW/8_proc/ && mv *.nc output_seymour/$NOW/8_proc/ &&

for filename in *.json; do mv "$filename" "output_seymour/$NOW/MPI_${NOW}_${filename#????}"; done;

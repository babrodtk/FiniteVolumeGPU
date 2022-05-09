#!/bin/bash
module purge
module load git/2.21.0 hdf5/1.10.5-gcc cuda/10.1

conda activate ShallowWaterGPU_HPC

python mpiTesting.py


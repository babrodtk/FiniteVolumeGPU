#!/bin/bash
module purge
module load git/2.21.0 hdf5/1.10.5-gcc cuda/10.1 conda/production

activate ShallowWaterGPU_HPC

/modules/centos7/conda/Feb2021/bin/python3 mpiTesting.py


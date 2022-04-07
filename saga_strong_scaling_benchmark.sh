#!/bin/bash

# one node: 1-4 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=1024,NY=1024 saga_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=1024,NY=512 saga_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=1024,NY=341 saga_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=512,NY=512 saga_strong_scaling_benchmark.job

# 2-4 nodes: 1 GPUs per node
sbatch --nodes=2 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=1024,NY=512 saga_strong_scaling_benchmark.job
sbatch --nodes=3 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=1024,NY=341 saga_strong_scaling_benchmark.job 
sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=512,NY=512 saga_strong_scaling_benchmark.job


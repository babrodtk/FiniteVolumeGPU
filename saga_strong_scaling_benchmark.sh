#!/bin/bash

# one node: 1-4 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=8192 saga_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=8192,NY=4096 saga_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=8192,NY=2731 saga_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=8192,NY=2048 saga_scaling_benchmark.job

# 2-4 nodes: 1 GPUs per node
sbatch --nodes=2 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=4096 saga_scaling_benchmark.job
sbatch --nodes=3 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=2731 saga_scaling_benchmark.job 
sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=2048 saga_scaling_benchmark.job

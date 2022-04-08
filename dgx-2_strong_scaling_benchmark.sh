#!/bin/bash

# one node: 1-8 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=4096,NY=4096 dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=4096,NY=2048 dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=4096,NY=1365 dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=4096,NY=1024 dgx-2_strong_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=4096,NY=819 dgx-2_strong_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=4096,NY=683 dgx-2_strong_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=4096,NY=585 dgx-2_strong_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=4096,NY=512 dgx-2_strong_scaling_benchmark.job

#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%dT%H%M%S")

# one node: 1-8 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=4096,NY=4096,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=4096,NY=2048,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=4096,NY=1365,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=4096,NY=1024,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=4096,NY=819,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=4096,NY=683,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=4096,NY=585,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job

sbatch --nodes=1 --gpus-per-node=9 --ntasks-per-node=9 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=10 --ntasks-per-node=10 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=11 --ntasks-per-node=11 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=12 --ntasks-per-node=12 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=13 --ntasks-per-node=13 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=14 --ntasks-per-node=14 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=15 --ntasks-per-node=15 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=16 --ntasks-per-node=16 --export=ALL,NX=4096,NY=512,NOW=$TIMESTAMP dgx-2_strong_scaling_benchmark.job

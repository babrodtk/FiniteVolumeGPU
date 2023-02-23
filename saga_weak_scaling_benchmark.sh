#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%dT%H%M%S")

# one node: 1-4 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 1 ranks
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 2 ranks
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 3 ranks
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks

# 2-4 nodes: 1 GPUs per node
sbatch --nodes=2 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 2 ranks
sbatch --nodes=3 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 3 ranks
sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks

## one node: 1-4 GPUs
#sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 1 ranks
#sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 2 ranks
#sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 3 ranks
#sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks

## 2-4 nodes: 1 GPUs per node
#sbatch --nodes=2 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 2 ranks
#sbatch --nodes=3 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 3 ranks
#sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=12288,NY=12288,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks
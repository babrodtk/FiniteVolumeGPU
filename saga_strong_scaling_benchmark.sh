#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%dT%H%M%S")

# one node: 1–4 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP saga_scaling_benchmark.job # 1 ranks
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=20480,NY=10240,NOW=$TIMESTAMP saga_scaling_benchmark.job # 2 ranks
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=20480,NY=6826,NOW=$TIMESTAMP saga_scaling_benchmark.job # 3 ranks
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=20480,NY=5120,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks

# 4 nodes: 1–4 GPUs per node
sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=5120,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks
sbatch --nodes=4 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=20480,NY=2560,NOW=$TIMESTAMP saga_scaling_benchmark.job # 8 ranks
sbatch --nodes=4 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=20480,NY=1706,NOW=$TIMESTAMP saga_scaling_benchmark.job # 12 ranks
sbatch --nodes=4 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=20480,NY=1280,NOW=$TIMESTAMP saga_scaling_benchmark.job # 16 ranks

# 4 nodes: 1–4 GPUs per node
sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=40960,NY=10240,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks
sbatch --nodes=4 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=40960,NY=5120,NOW=$TIMESTAMP saga_scaling_benchmark.job # 8 ranks
sbatch --nodes=4 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=40960,NY=3413,NOW=$TIMESTAMP saga_scaling_benchmark.job # 12 ranks
sbatch --nodes=4 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=40960,NY=2560,NOW=$TIMESTAMP saga_scaling_benchmark.job # 16 ranks

## one node: 1–4 GPUs
#sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=24576,NY=6144,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks
#
## 4 nodes: 1–4 GPUs per node
#sbatch --nodes=4 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=24576,NY=6144,NOW=$TIMESTAMP saga_scaling_benchmark.job # 4 ranks
#sbatch --nodes=4 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=24576,NY=3072,NOW=$TIMESTAMP saga_scaling_benchmark.job # 8 ranks
#sbatch --nodes=4 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=24576,NY=2048,NOW=$TIMESTAMP saga_scaling_benchmark.job # 12 ranks
#sbatch --nodes=4 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=24576,NY=1536,NOW=$TIMESTAMP saga_scaling_benchmark.job # 16 ranks

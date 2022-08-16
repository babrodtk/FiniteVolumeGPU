#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%dT%H%M%S")

# one node: 1-16 GPUs
#sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job

# one node: 1-8 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=20480,NY=20480,NOW=$TIMESTAMP hgx_scaling_benchmark.job

#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%dT%H%M%S")

# one node: 1-8 GPUs
#sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=8192,NY=4096,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=8192,NY=2731,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=8192,NY=2048,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=8192,NY=1638,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=8192,NY=1365,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=8192,NY=1170,NOW=$TIMESTAMP hgx_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=8192,NY=1024,NOW=$TIMESTAMP hgx_scaling_benchmark.job

# one node: 4-8 GPUs
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=41984,NY=10496,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=41984,NY=8396,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=41984,NY=6997,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=41984,NY=5997,NOW=$TIMESTAMP hgx_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=41984,NY=5248,NOW=$TIMESTAMP hgx_scaling_benchmark.job

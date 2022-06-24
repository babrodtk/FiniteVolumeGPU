#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%dT%H%M%S")

# one node: 1-16 GPUs
#sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=8192,NY=8192,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=8192,NY=4096,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=8192,NY=2731,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=8192,NY=2048,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=8192,NY=1638,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=8192,NY=1365,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=8192,NY=1170,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=8192,NY=1024,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#
#sbatch --nodes=1 --gpus-per-node=9 --ntasks-per-node=9 --export=ALL,NX=8192,NY=910,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=10 --ntasks-per-node=10 --export=ALL,NX=8192,NY=819,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=11 --ntasks-per-node=11 --export=ALL,NX=8192,NY=745,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=12 --ntasks-per-node=12 --export=ALL,NX=8192,NY=683,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=13 --ntasks-per-node=13 --export=ALL,NX=8192,NY=630,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=14 --ntasks-per-node=14 --export=ALL,NX=8192,NY=585,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=15 --ntasks-per-node=15 --export=ALL,NX=8192,NY=546,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=16 --ntasks-per-node=16 --export=ALL,NX=8192,NY=512,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job

# one node: 4-16 GPUs
#sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=41984,NY=10496,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=41984,NY=8396,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=41984,NY=6997,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=41984,NY=5997,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=41984,NY=5248,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#
#sbatch --nodes=1 --gpus-per-node=9 --ntasks-per-node=9 --export=ALL,NX=41984,NY=4664,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=10 --ntasks-per-node=10 --export=ALL,NX=41984,NY=4198,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=11 --ntasks-per-node=11 --export=ALL,NX=41984,NY=3816,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=12 --ntasks-per-node=12 --export=ALL,NX=41984,NY=3498,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=13 --ntasks-per-node=13 --export=ALL,NX=41984,NY=3229,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=14 --ntasks-per-node=14 --export=ALL,NX=41984,NY=2998,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=15 --ntasks-per-node=15 --export=ALL,NX=41984,NY=2798,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
#sbatch --nodes=1 --gpus-per-node=16 --ntasks-per-node=16 --export=ALL,NX=41984,NY=2624,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job

# one node: 1-16 GPUs
sbatch --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --export=ALL,NX=22528,NY=22528,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=2 --ntasks-per-node=2 --export=ALL,NX=22528,NY=11264,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=3 --ntasks-per-node=3 --export=ALL,NX=22528,NY=7509,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=22528,NY=5632,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=22528,NY=4505,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=22528,NY=3754,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=22528,NY=3218,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=22528,NY=2816,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job

sbatch --nodes=1 --gpus-per-node=9 --ntasks-per-node=9 --export=ALL,NX=22528,NY=2503,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=10 --ntasks-per-node=10 --export=ALL,NX=22528,NY=2252,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=11 --ntasks-per-node=11 --export=ALL,NX=22528,NY=2048,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=12 --ntasks-per-node=12 --export=ALL,NX=22528,NY=1877,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=13 --ntasks-per-node=13 --export=ALL,NX=22528,NY=1732,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=14 --ntasks-per-node=14 --export=ALL,NX=22528,NY=1609,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=15 --ntasks-per-node=15 --export=ALL,NX=22528,NY=1501,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=16 --ntasks-per-node=16 --export=ALL,NX=22528,NY=1408,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job

# one node: 4-16 GPUs
sbatch --nodes=1 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL,NX=45056,NY=11264,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=5 --ntasks-per-node=5 --export=ALL,NX=45056,NY=8396,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=6 --ntasks-per-node=6 --export=ALL,NX=45056,NY=6997,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=7 --ntasks-per-node=7 --export=ALL,NX=45056,NY=5997,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=8 --ntasks-per-node=8 --export=ALL,NX=45056,NY=5248,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job

sbatch --nodes=1 --gpus-per-node=9 --ntasks-per-node=9 --export=ALL,NX=45056,NY=4664,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=10 --ntasks-per-node=10 --export=ALL,NX=45056,NY=4198,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=11 --ntasks-per-node=11 --export=ALL,NX=45056,NY=3816,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=12 --ntasks-per-node=12 --export=ALL,NX=45056,NY=3498,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=13 --ntasks-per-node=13 --export=ALL,NX=45056,NY=3229,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=14 --ntasks-per-node=14 --export=ALL,NX=45056,NY=2998,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=15 --ntasks-per-node=15 --export=ALL,NX=45056,NY=2798,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job
sbatch --nodes=1 --gpus-per-node=16 --ntasks-per-node=16 --export=ALL,NX=45056,NY=2624,NOW=$TIMESTAMP dgx-2_scaling_benchmark.job

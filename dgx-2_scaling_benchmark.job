#!/bin/bash
# See http://wiki.ex3.simula.no before changing the values below
#SBATCH -p dgx2q                   # partition (GPU queue)
#SBATCH -w g001                    # DGX-2 node
##SBATCH --gres=gpu:4               # number of V100's
#SBATCH -t 0-00:10                 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR
#SBATCH --reservation=martinls_17


# For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
# To enable it, please set the environment variable OMPI_MCA_opal_cuda_support=true before
# launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
# mpiexec --mca opal_cuda_support 1 ...
# 
# In addition, the UCX support is also built but disabled by default.
# To enable it, first install UCX (conda install -c conda-forge ucx). Then, set the environment
# variables OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" before launching your MPI processes.
# Equivalently, you can set the MCA parameters in the command line:
# mpiexec --mca pml ucx --mca osc ucx ...
# Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via UCX.
# Please consult UCX's documentation for detail.

ulimit -s 10240
module load slurm/20.02.7
module load cuda11.2/toolkit/11.2.2
module load openmpi4-cuda11.2-ofed50-gcc8/4.1.0

# Check how many gpu's your job got
#nvidia-smi

mkdir -p output_dgx-2/$NOW

## Copy input files to the work directory:
mkdir -p /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU
cp -r . /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU

# Run job
# (Assumes Miniconda is installed in user root dir.)
cd /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU
#mpirun --mca btl_openib_if_include mlx5_0 --mca btl_openib_warn_no_device_params_found 0 $HOME/miniconda3/envs/ShallowWaterGPU_HPC/bin/python3 mpiTesting.py -nx $NX -ny $NY --profile
#nsys profile -t nvtx,cuda mpirun -np  $SLURM_NTASKS numactl --cpunodebind=0 --localalloc $HOME/miniconda3/envs/ShallowWaterGPU_HPC/bin/python3 mpiTesting.py -nx $NX -ny $NY --profile
#mpirun -np $SLURM_NTASKS numactl --cpunodebind=0 --localalloc $HOME/miniconda3/envs/ShallowWaterGPU_HPC/bin/python3 mpiTesting.py -nx $NX -ny $NY --profile

export OMPI_MCA_opal_cuda_support=true
mpirun -np $SLURM_NTASKS $HOME/miniconda3/envs/ShallowWaterGPU_HPC/bin/python3 mpiTesting.py -nx $NX -ny $NY --profile

cd $HOME/src/ShallowWaterGPU

## Copy files from work directory:
# (NOTE: Copying is not performed if job fails!)
mkdir -p output_dgx-2/$NOW/$SLURM_JOB_ID
mv /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU/*.log ./output_dgx-2/$NOW/$SLURM_JOB_ID
mv /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU/*.nc ./output_dgx-2/$NOW/$SLURM_JOB_ID
mv /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU/*.json ./output_dgx-2/$NOW
mv /work/$USER/$SLURM_JOB_ID/ShallowWaterGPU/*.qdrep ./output_dgx-2/$NOW

rm -rf /work/$USER/$SLURM_JOB_ID

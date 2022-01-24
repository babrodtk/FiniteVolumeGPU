# ShallowWaterGPU

## Systems
Connection and run details for all potential benchmark systems.

### OsloMet 2 x Quadro RTX 6000 (VPN necessary)
Connect:  
ssh -AX ip-from-webpage  
  
For Jupyter Notebook:  
ssh -L 8888:localhost:80 ip-from-webpage  
(access localhost:8888 in browser and open terminal)  
conda activate ShallowWaterGPU_HPC  
(one time operation) conda install -c conda-forge ipyparallel  

### Simula DGX-2
Connect:  
ssh -YAC2 dnat.simula.no -p 60441 (and then ssh -Y g001 for direct login to DGX-2 box)

Example job script:  
dgx-2-test.job

Submit:  
sbatch dgx-2-test.job

### PPI 4 x P100 (VPN necessary)
Connect:  
ssh -AX gpu-01.ppi.met.no  
ssh -AX gpu-02.ppi.met.no  
ssh -AX gpu-03.ppi.met.no  
ssh -AX gpu-04.ppi.met.no

Submit:  
run_script_ppi.sh

### Saga 8 nodes with 4 x P100
Connect:  
ssh -AX saga.sigma2.no

Example job script:  
saga-test.job

Submit:  
sbatch saga-test.job

### Granada: 2 x RTX 2080 + 1 x RTX "new" + X x P100

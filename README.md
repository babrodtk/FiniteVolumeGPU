# ShallowWaterGPU

## Systems
Connection and run details for all potential benchmark systems.

### OsloMet ? (VPN necessary)
Connect:
ssh -AX <ip from webpage> (or Jupyter Notebook through https)

### Simula DGX-2
Connect:
ssh -YAC2 martinls@dnat.simula.no -p 60441 (and then ssh -Y g001 for direct login to DGX-2 box)

Example job script:
dgx-2-test.job

Submit:
sbatch dgx-2-test.job

### PPI (VPN necessary)
Connect: 
ssh -AX gpu-01.ppi.met.no
ssh -AX gpu-02.ppi.met.no
ssh -AX gpu-03.ppi.met.no
ssh -AX gpu-04.ppi.met.no

Submit:
run_script_ppi.sh

### Saga
Connect:
ssh -AX saga.sigma2.no

Example job script:
saga-test.job

Submit:
sbatch saga-test.job

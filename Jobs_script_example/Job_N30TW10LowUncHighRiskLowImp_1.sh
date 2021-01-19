#!/bin/bash

#Submit script with: sbatch thefilename
#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -p batch   # partition(s)
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH -J N30TW10LowUncHighRiskLowImp_1   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
cd /homes/up201202787/BRKGA_Asset_GRID_Laplace/Clustered_LowUncHighRiskLowImp
./main data/N30TW10LowUncHighRiskLowImp_1_1.txt 0 0 0 50 50 10000
./main data/N30TW10LowUncHighRiskLowImp_1_1.txt 1 0 0 50 50 10000
./main data/N30TW10LowUncHighRiskLowImp_1_1.txt 0 1 0 50 50 10000
./main data/N30TW10LowUncHighRiskLowImp_1_1.txt 0 0 1 50 50 10000
./main data/N30TW10LowUncHighRiskLowImp_1_1.txt 1 1 1 50 50 10000
# End of bash script
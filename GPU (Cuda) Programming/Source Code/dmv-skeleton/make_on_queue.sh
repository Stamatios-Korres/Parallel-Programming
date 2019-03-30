#!/bin/bash

## Give the Job a descriptive name
#PBS -N makejob

## Limit memory, runtime etc.
#PBS -l walltime=00:01:00


## Output and error files
#PBS -o make.out
#PBS -e make.err

## How many machines should we get?
#PBS -l nodes=1:ppn=24:cuda

## Start 
## Run make in the src folder (modify properly)

cd /home/parallel/parlab11/4-timos/dmv-skeleton/
module load gcc/4.8.2
make


#!/bin/bash
#
## run_dmv.sh -- Run DMV in GPU systems.
##
## This is an example script for submitting to the Torque system your
## experiments. You can freely change the script as you like. Please respect the
## `walltime' attribute. The full experiments set won't take more than 4 hours.
##
## Please remember to compile your code with `make DEBUG=0' before
## submitting. If you plan to use this script, we recommend you to enable only
## the GPU kernels to avoid unnecessary executions of the serial and OpenMP
## version of the code wasting your time. Use similar scripts with just the
## required executions for these versions.
##
## Copyright (C) 2012, Computing Systems Laboratory (CSLab)
##

#PBS -o dmv_gpu_2D.out
#PBS -e dmv_gpu_2D.err
#PBS -l walltime=06:00:00
#PBS -l nodes=1:ppn=24:cuda

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

gpu_kernels="0 1 2"
problem_sizes="1 2 4 7 8 14 16" # in Kibi's (x1024)
block_sizes="$(seq 16 16 512)"
gpu_prog="./dmv_main"
cd /home/parallel/parlab11/4-timos/dmv-skeleton/
echo "Benchmark started on $(date) in $(hostname)"
for k in $gpu_kernels; do
    for p in $problem_sizes; do
        for b in $block_sizes; do
        	GPU_KERNEL=$k GPU_BLOCK_SIZE=$b $gpu_prog $((p*1024))
       done
    done
done
echo "Benchmark ended on $(date) in $(hostname)"

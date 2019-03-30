/*
 *  dmv_gpu.cu -- Template for DMV GPU kernels
 *
 *  Copyright (C) 2010-2013, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2010-2013, Vasileios Karakasis
 */ 
#include <stdio.h>
#include "dmv.h"

/*
 *  Utility function to get the thread ID within the
 *  global working space.
 */ 
__device__ int get_global_tid()
{
	return (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x  *blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
}

/*
 *  Utility function to get the thread ID within the
 *  local/block working space.
 */ 
__device__ int get_local_tid()
{
	return blockDim.x*threadIdx.y + threadIdx.x;
	
	
}

__global__ void dmv_gpu_naive(const value_t *a, const value_t *x, value_t *y, size_t n)
{
	value_t sum=0;
	int j;
	int row = blockIdx.x*blockDim.x+threadIdx.x;
	for(j=0;j<n;j++){
     		sum += a[row*n+j]*x[j];
  	}
   	y[row]=sum;
}

__global__ void dmv_gpu_coalesced(const value_t *a, const value_t *x,
                                  value_t *y, size_t n)
{
	int row = blockIdx.x*blockDim.x+threadIdx.x;
        value_t sum=0;
        int j;
        for(j=0;j<n;j++){
                sum += a[row+j*n]*x[j];
        }
        y[row]=sum;
}

__global__ void dmv_gpu_shmem(const value_t *a, const value_t *x, value_t *y,size_t n)
{
	extern __shared__ value_t sharing[];
	int global = blockIdx.x*blockDim.x+threadIdx.x;
	int i,j;
	value_t sum = 0;
	for(i=0;i<gridDim.x;i++){
		__syncthreads();
		sharing[threadIdx.x] = x[threadIdx.x + i*blockDim.x];
		__syncthreads();
		for(j=0;j<blockDim.x;j++){
				sum = sum + (a[ global + j*n + (blockDim.x*n)*i ] * sharing[j]);
		}
	}
        y[global]=sum;
}

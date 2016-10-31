// Using different memory sapces in CUDA

#include <stdio.h>

/*
 * Using local memory
 */

// a __device__ or __global__ function runs on the GPU
__global__ void use_local_memory_GPU(float in)
{
	float f; // varible "f" is in local memory and private to each thread
	f = in;
}

/*
 * Using global memory
 */

// a __global__ function runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float* array)
{
	// "array" is a pointer into global memory on the device
	array[threadIdx.x] = 2.0f * (float)threadIdx.x;
}

int main(int argc, char** argv)
{
	/* 
	 * First, call a kernel that shows using local memory
	 */
	use_local_memory_GPU<<<1, 128>>>(2.0f);
	
	return 0;
}

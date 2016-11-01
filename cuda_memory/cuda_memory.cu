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
	
	/*
	 * Next, call a kernel that shows using global memory
	 */
	float h_arr[128]; // convention: h_ variables live on host
	float *d_arr;	  // convention: d_ variables live on device (GPU global mem)  
	
	// allocate global memory on the device, place result in "d_arr"
	cudaMalloc((void**)&d_arr, sizeof(float) * 128);
	// now copy data from host memory to device memory
	cudaMemcpy((void *)d_arr, (void*)h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
	// launch the kernel 
	use_global_memory_GPU<<<1, 128>>>(d_arr);
	// copy the modified array back to the host, overwriting contents of h_arr
	cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
	// ... do other stuff ...
	for (int i = 0; i < 128; ++i)
	{
		printf("%f", h_arr[i]);
		printf((i % 4 != 3) ? "\t" : "\n");
	}
	return 0;
}

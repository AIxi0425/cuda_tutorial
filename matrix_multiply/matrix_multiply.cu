#include <stdio.h>

__host__ void matrix_multiply_CPU(int *h_p, int *h_q, int *h_m, int width)
{
	for (int i = 0; i < width; i++)
		for (int j = 0; j< width; j++)
			for (int k = 0; k < width; k++)
			{
				int tmp = h_p[i*width+k] * h_q[k*width+j];
				h_m[i*width+j] += tmp;
			}
}

// __global__ runs on the GPU & can be called from host
// __global__ must return void
__global__ void matrix_multiply_GPU(int *d_p, int *d_q, int *d_m, int width)
{
//	int row = threadIdx.y + blockIdx.y * blockDim.y;
//	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y;
	int col = threadIdx.x;
	
	for (int k = 0; k < width; k++)
	{
		int tmp = d_p[row*width+k] * d_q[k*width+col]; 
		d_m[row*width+col] += tmp;			
	}

}

int main()
{
	int *h_p, *h_q, *h_m;
	const int width = 5;

	h_p = (int*)malloc(sizeof(int) * width * width);
	h_q = (int*)malloc(sizeof(int) * width * width);
	h_m = (int*)malloc(sizeof(int) * width * width);

	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
			h_p[i*width+j] = i*width + j;

	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
			h_q[i*width+j] = i*width + j;
	
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
			h_m[i*width+j] = 0;

	matrix_multiply_CPU(h_p, h_q, h_m, width);
	printf("The result of matrix_multiply_CPU:\n");
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
		{
			printf("%d",h_m[i*width+j]);
			printf((j != width-1) ? "\t" : "\n");
		}

	printf("---------------------\n");

	// decalar GPU memory pointers
	int *d_p, *d_q, *d_m;

	// allocate GPU memory 
	cudaMalloc((void **)&d_p, sizeof(int) * width * width);
	cudaMalloc((void **)&d_q, sizeof(int) * width * width);
	cudaMalloc((void **)&d_m, sizeof(int) * width * width);

	// zero out GPU memory
	cudaMemset((void *)d_m, 0, sizeof(int) * width * width);

	// transfer the matrix to the GPU
	cudaMemcpy((void *)d_p, (void *)h_p, sizeof(int) * width * width, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_q, (void *)h_q, sizeof(int) * width * width, cudaMemcpyHostToDevice);

	// 
	const dim3 cat(1, 1);
	const dim3 dog(width, width, 1);

	// launch the kernel
	matrix_multiply_GPU<<<cat, dog>>>(d_p, d_q, d_m, width);

	// copy back the matrix from GPU to the CPU
	cudaMemcpy((void *)h_m, (void *)d_m, sizeof(int) * width * width, cudaMemcpyDeviceToHost);

	// print the matrix
	printf("The result of matrix_multiply_GPU:\n");
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
		{
			printf("%d",h_m[i*width+j]);
			printf((j != width-1) ? "\t" : "\n");
		}
	
	// free memory allocation
	free(h_p);
	free(h_q);
	free(h_m);
	cudaFree(d_p);
	cudaFree(d_q);
	cudaFree(d_m);

	return 0;
}

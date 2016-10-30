#include <stdio.h>

__global__ void hello()
{
	printf("hello, My gridDim is %d, %d, %d\n", gridDim.x, gridDim.y, gridDim.z);
	printf("hello, My blockDim is %d, %d, %d\n", blockDim.x, blockDim.y, blockDim.z);
}

int main(int argc, char** argv)
{
	dim3 cat(1, 2, 3);
	dim3 dog(2, 1, 2);
	
	hello<<<cat, dog>>>();
	
	cudaDeviceSynchronize();
	
	printf("That's all");

	return 0;
}

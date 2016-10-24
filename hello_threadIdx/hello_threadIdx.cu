#include <stdio.h>


#define NUM_BLOCKS 1
#define BLOCKS_WIDTH 256

__global__ void hello()
{
	printf("Hello world! I am thread %d\n", threadIdx.x);
}

int main(int argc, char** argv)
{
	// launch the kernel
	hello<<<NUM_BLOCKS, BLOCKS_WIDTH>>>();
	
	// force the printf()s to flush
	cudaDeviceSynchronize();

	printf("That's all!\n");

	return 0;
}

#include <stdio.h>


#define NUM_BLOCKS 16
#define BLOCK_NUM 1

__global__ void hello()
{
	printf("hello world! I am a thread in block %d\n", blockIdx.x);
}

int main(int argc, char** argv)
{
	// launch the kernel
	hello<<<NUM_BLOCKS, BLOCK_NUM>>>();
	
	// force the cout to flush
	cudaDeviceSynchronize();

	printf("That's all!\n");

	return 0;
}

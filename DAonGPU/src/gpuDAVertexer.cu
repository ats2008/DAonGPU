#include "gpuDAVertexer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

__global__ void demoKernel()
{
	printf("On Devise !! %d %d \n",threadIdx.x,blockIdx.x);
}

ZVertexSoA * gpuDAVertexer::DAVertexer::makeAsync(ZTrackSoA * track)
{
  
  demoKernel<<<2,2>>>();
 cudaThreadSynchronize(); 
  return nullptr;
 
}


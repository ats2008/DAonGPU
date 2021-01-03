#include "gpuDAVertexer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

__global__ void demoKernel(ZTrackSoA * tracks,int n)
{
	if(!tracks) printf("null as trks");
        int idx =threadIdx.x + blockIdx.x*blockDim.x;
	//printf("HI HI !! in device %d  idx = %d %f \n",n,idx,tracks[15].pt[0]);
	if (idx<n)
	{
	 	printf("On Devise !! [%d + %d * %d]  : track[%d].pt[0] = %f \n",threadIdx.x,blockIdx.x,blockDim.x,idx,tracks[idx].pt[0]);
	}
	else
	{
		printf("On Devise !! [%d + %d * %d] = %d\n ",threadIdx.x,blockIdx.x,blockDim.x,idx);
	}
}

ZVertexSoA * gpuDAVertexer::DAVertexer::makeAsync(ZTrackSoA * tracks,int n)
{
  printf("\n in the makeAsync n = %d \n",n);	  
  demoKernel<<<2,10>>>(tracks,n);
  cudaDeviceSynchronize(); 
  return nullptr;
}


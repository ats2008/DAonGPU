#include "gpuDAVertexer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdio.h"

namespace gpuDAVertexer{

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

__global__ void loadTracks(ZTrackSoA * tracks,Workspace * wrkspace)
{

	auto idx= blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < tracks->ntrks)
	{
		wrkspace->zt[idx]=tracks->zt[idx];
		wrkspace->dz2[idx]=tracks->dz2[idx];
		wrkspace->pi[idx]=1.0;
	}

}

__global__ void calculateT0(Workspace * wrkspace)
{
	if(threadIdx.x==0)
	printf("In the calculateT0 kernel \n");
}

// device functions might also be made inline , have to check if we will get any performance inprovements in this
// The calculation of Z, the Eik calculation has  space complexity of numTracks*numVertices
// In the original DA code they have arranged tracks in the acending Z and only the tracks which are close to a vertex goes into the ccalculation
// the farer tracks will only contibute very less since its supressed by exp (-Eik ) , we may have to also incorporate it after a basic working code is ready.


__device__ void updateTrackToVertexProbablilities()
{
	if(threadIdx.x==0)
	printf("In the updateTrackToVertexProbablilities\n");
}

__device__ void updateVertexPositions()
{
	if(threadIdx.x==0)
	printf("In the updateVertexPositions\n");
}

__device__ void updateVertexWeights()
{
	if(threadIdx.x==0)
	printf("In the updateVertexWeights\n");
}

__device__  void updateClusterCriticalTemperatures()
{
	if(threadIdx.x==0)
	printf("In the updateClusterCriticalTemperatures\n");
}
__device__ void checkAndSplitClusters()
{
	if(threadIdx.x==0)
	printf("In the checkAndSplitClusters\n");

}

__device__ void checkAndMergeClusters()
{
	if(threadIdx.x==0)
	printf("In the checkAndMergeClusters \n");
}

__global__ void dynamicSplittingPhase(Workspace * wrkspace)
{
	auto &workspace = *wrkspace;
	int i=0;
	while(i<2 /*workspace.beta < workspace.betaSplitMax */)
	{
	
   // this could be avoided if we could store a sequnce of betas in the worspace precomputed
		if(threadIdx.x==0)
		  {
		  	workspace.beta*=workspace.betaFactor;
			i+=1;
			printf("at dynamicSplittingPhase with i = %d  zt[0] = %f \n ",i,workspace.zt[0]);
		  }
		 else i++;
		updateTrackToVertexProbablilities();
		__syncthreads();
		updateVertexPositions();
		__syncthreads();
		updateVertexWeights();
		__syncthreads();
		updateClusterCriticalTemperatures();
		__syncthreads();
		checkAndSplitClusters();
	}

	checkAndMergeClusters();
	return ;
}

__global__ void vertexAssignmentPhase(Workspace * wrkspace)
{
	auto &workspace =*wrkspace;

	// this may require some restructuring
	// there is a possibility of this to paralized with each block taking up thermalization of an individual vertex and moving the checkAndMergeClusters() to another __global__ kernel
	int i=0;
	while( i<2  /*workspace.beta < workspace.betaMax*/ )
	{
		updateTrackToVertexProbablilities();
		__syncthreads();
		updateVertexPositions();
		__syncthreads();
		updateVertexWeights();
		__syncthreads();
		checkAndMergeClusters();
		__syncthreads();
		
   // this could be avoided if we could store a sequnce of betas in the worspace precomputed
		if(threadIdx.x==0){
		   workspace.beta*=workspace.betaFactor;
		   i++;
		   printf("at vertexAssignmentPhase with i = %d , dz2[0] = %f \n",i,workspace.dz2[0]);
		   }
		else i++;
		__syncthreads();
	
	}

	return;
}

ZVertexSoA * DAVertexer::makeAsync(ZTrackSoA * tracks,int n)
{
 
	 Workspace *wrkspace;
	 cudaMalloc(&wrkspace,sizeof(Workspace));
	
	 //demoKernel<<<2,10>>>(tracks,n);
	 //cudaDeviceSynchronize(); 
	 
	 auto numberOfThreads = 128;
	 auto numberOfBlocks  = (MAXTRACKS/numberOfThreads) + 1;
	
	 loadTracks<<<numberOfBlocks,numberOfThreads>>>(tracks,wrkspace);
	 cudaDeviceSynchronize(); 
	 printf("\n");
	 dynamicSplittingPhase<<<1,1024>>>(wrkspace);
	 cudaDeviceSynchronize(); 
	 printf("\n");
	 vertexAssignmentPhase<<<1,102>>>(wrkspace);
	 printf("\n");
	 
	 //printf(cudaGetErrorName(cudaGetLastError()));
	 cudaDeviceSynchronize(); 
	 printf("\n");
	
	 return nullptr;

}

}

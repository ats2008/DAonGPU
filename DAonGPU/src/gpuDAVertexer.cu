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

__global__ void initializeWorspace(Workspace * wrkspace)
{

	auto idx= blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < MAXTRACKS*MAXVTX)
	{
		wrkspace->pik[idx]=1.0;
		wrkspace->pik_numer[idx]=0.0;
		wrkspace->pik_denom[idx]=0.0;
		
		wrkspace->zk_delta[idx]=1e8;
		wrkspace->zk_numer[idx]=0.0;
		wrkspace->zk_denom[idx]=0.0;
	}
	if(idx <MAXVTX)
	{
		wrkspace->zVtx[idx]=1e9;
	}
	if(idx==0)
	{
		wrkspace->nVertex=0;
		wrkspace->betaFactor=2.5;
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
	if(idx==0)
	{
		if(tracks->ntrks % 2 and false)	
		{
	 	  wrkspace->zt[idx]=0.0;
		  wrkspace->dz2[idx]=1e9;
		  wrkspace->pi[idx]=0.0;
                  wrkspace->nTracks = tracks->ntrks+1; 
		}
		else
		 {
			wrkspace->nTracks=tracks->ntrks;
		 }
	}

}

	/// ==================================================//

__global__ void sumBlock_with_shfl_down_gid(float *in, float *out, int blockSize)
{ 
	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) 
	{        	 
	  out[gid] +=  __shfl_down_sync(0xffffffff, out[gid], offset);     
	}

 
}

__global__ void sumBlock_with_loop(float *in, float *out,int numVertices, int N)
{ 
	int tid = threadIdx.x; 
	int off ;
	out[tid]=0.0;	
	for (int offset = 0 ; offset<numVertices; offset++ )  
	{        	  
	  off = N*offset ;
	  out[tid] += in[tid+off];     
	}
//	printf("out[%d] = %f \n",tid,out[tid]);
}
__global__ void kernel_findFreeEnergyPartA(float *FEnergyA,float * zi, float *zVtx,float* sig,float beta ,int CurrentVtx,int N )
{
    int idx = threadIdx.x; 
    int bid = blockIdx.x; // block Id
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*CurrentVtx; //  nTracks * nVertex    

    if (gid < TotalSize) {
       FEnergyA[gid] =  expf( -beta*((zi[idx]-zVtx[bid])*(zi[idx]-zVtx[bid])/sig[idx] ));
     printf("gid = %d , dz = %f - %f = %f , FEnergyA[gid] = %f\n",gid,zi[idx],zVtx[bid],zi[idx]-zVtx[bid], FEnergyA[gid]);
    }
}
__global__ void kernel_findFreeEnergyPartB(float * FEnergyA, float beta, int currVtxCount,int N)
{

	auto fEnergy=0.0;
	for(int i=0;i<N;i++)
	{
	 auto asum=0.0;
	 for(int j=0;j<currVtxCount;j++)
		{

			asum+=FEnergyA[i+j*N];
		}
	printf("( %d , %f ,%f )",i,asum,beta);	
		fEnergy-=logf(asum >1e-20 ? asum : 1.0 )/beta;
	
	}
	printf("\n$FE,%f,%f\n",beta,fEnergy);
}



__global__ void kernel_p_ik_num( float *p_ik, float *z_i, float *z_k0,   float *sig, float beta, int N, int numberOfvertex )
{

    int idx = threadIdx.x; 
    int bid = blockIdx.x; // block Id
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    

    if (gid < TotalSize) {
       p_ik[gid] =  expf( -beta*(((z_i[idx]-z_k0[bid])*(z_i[idx]-z_k0[bid]))/(sig[idx]*sig[idx]*sig[idx]*sig[idx])) );
  //  printf("gid = %d , dz = %f - %f = %f , pik = %f\n",gid,z_i[idx],z_k0[bid],z_i[idx]-z_k0[bid], p_ik[gid]);
    }

}

__global__ void kernel_p_ik( float *p_ik, float *p_ik_den, int N, int numberOfvertex )
{
    
    int idx = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; 

    if (gid < TotalSize) 
    { 

	 //auto oldval=p_ik[gid];
   	 if (p_ik_den[idx] > 1.e-45) 
   	 {   
   	     p_ik[gid] =  p_ik[gid]/p_ik_den[idx] ;
   	 }
   	 else
   	 {
   	     p_ik[gid] =  0.000 ;     
   	 }

    //printf("pik[%d] = pik_[%d] / p_ik_den[%d] = %f/ %f = %f\n",\\
    		gid,gid,idx,oldval,p_ik_den[idx],p_ik[gid]);
    }

}


__global__ void kernel_z_ik_num( float *p_ik, float *z_ik_num, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
    int idx = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
    if (gid < TotalSize) { 
        z_ik_num[gid] = p_i[idx]*p_ik[gid]*z_i[idx]/(sig[idx]*sig[idx]); 
	//printf("z_ik_num[%d] = %f ,idx  = %d , z_i[idx] = %f , p_i[idx]  = %f ,	sig[idx] = %f \n",gid,z_ik_num[gid],idx,z_i[idx],p_i[idx],sig[idx]);
    }
}


__global__ void kernel_z_ik_den( float *p_ik, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
    int idx = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
    if (gid < TotalSize) {  
        z_ik_den[gid] = p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
	//printf("z_ik_den[%d] = %f ,idx  = %d , p_i[idx]  = %f ,	sig[idx] = %f \n",gid,z_ik_den[gid],idx,p_i[idx],sig[idx]);
    }

}

__device__ void  kernel_z_ik(float * zk_numer,float * zk_denom,float * zDelta,float* zVtx,int ntraks,int currVtxCount )
{
	if(threadIdx.x < currVtxCount)
	{
	  auto newZ=zk_numer[threadIdx.x*ntraks]/(1e-20 + zk_denom[threadIdx.x*ntraks]);
	  zDelta[threadIdx.x] = abs(zVtx[threadIdx.x] - newZ);
	  zVtx[threadIdx.x]   = newZ;
	  printf("setting Z[%d] = %f ,delta[%d] = %f ,numer = %f , deno = %f \n",\\
			threadIdx.x,zVtx[threadIdx.x],threadIdx.x,zDelta[threadIdx.x], zk_numer[threadIdx.x*ntraks],zk_denom[threadIdx.x]);
	}
}



__global__ void kernel_T0_num( float *T_num, float *z_i, float *zVtx, float *p, float *sig, int N,int currVtxCount )
{
	auto tid = threadIdx.x; 
	auto idx = threadIdx.x; 
	for(auto i=0;i<currVtxCount;i++)
	{
		idx+=N*i;
       		T_num[idx] = p[tid]*((z_i[tid]-zVtx[i])*(z_i[tid]-zVtx[i]))/(sig[tid]*sig[tid]); 
 //	printf("tid = %d, p[tid] =%f , z_i[tid] = %f ,zVtx[%d] =%f ,sig[tid] =%f , Tnum[%d] = %f \n",tid,p[tid],z_i[tid],i,zVtx[i],sig[tid],idx,T_num[idx]);
	}

	
}

__device__ void  kernel_tc_k(float * tc_numer,float * tc_denom,float* tc,int ntraks,int currVtxCount )
{
	if(threadIdx.x < currVtxCount)
	{
	  tc[threadIdx.x] = 2.0*tc_numer[threadIdx.x*ntraks]/(1e-20 + tc_denom[threadIdx.x*ntraks]);
	  //printf("\n setting tc[%d] = %f , numer = %f , deno = %f \n",\\
			threadIdx.x,tc[threadIdx.x], tc_numer[threadIdx.x*ntraks],tc_denom[threadIdx.x]);
	}
}
	/// =================================================//

__global__ void calculateT0(Workspace * wrkspace)
{
	if(threadIdx.x==0)
	printf("In the calculateT0 kernel \n");
}

// device functions might also be made inline , have to check if we will get any performance inprovements in this
// The calculation of Z, the Eik calculation has  space complexity of numTracks*numVertices
// In the original DA code they have arranged tracks in the acending Z and only the tracks which are close to a vertex goes into the ccalculation
// the farer tracks will only contibute very less since its supressed by exp (-Eik ) , we may have to also incorporate it after a basic working code is ready.


__global__ void initializeDAvertexReco( Workspace *wrkspace  )
{
	
	auto N=wrkspace->nTracks;
	printf("N  = %d \n",N);
	auto CurrentNvetex = 1;
	//      >>>>>>>>>KERNELs for ZVtx Update<<<<<<<<<  
	kernel_z_ik_num<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_numer, wrkspace->pi,wrkspace->zt,wrkspace->dz2, N, CurrentNvetex);
	cudaDeviceSynchronize();
	kernel_z_ik_den<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_denom, wrkspace->pi, wrkspace->zt, wrkspace->dz2, N, CurrentNvetex); 
	cudaDeviceSynchronize();
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_numer, wrkspace->zk_numer, N); 
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_denom, wrkspace->zk_denom, N);  	
	cudaDeviceSynchronize();
	kernel_z_ik(wrkspace->zk_numer, wrkspace->zk_denom,wrkspace->zk_delta ,wrkspace->zVtx, N, CurrentNvetex);  

	wrkspace->nVertex=1;
 	 	
	//      >>>>>>>>>KERNEL for T finding <<<<<<<<<	
	kernel_T0_num<<<1, N>>>(wrkspace->tc_numer,wrkspace->zt,\\
					wrkspace->zVtx,wrkspace->pi ,wrkspace->dz2,\\
					N,CurrentNvetex);
	cudaDeviceSynchronize();
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->tc_numer,wrkspace->tc_numer,N);
	// note that the denominator for Zk and Tc_k are same				
	cudaDeviceSynchronize();
	kernel_tc_k(wrkspace->tc_numer,wrkspace->zk_denom,wrkspace->tc,N,CurrentNvetex);
	cudaDeviceSynchronize();
	wrkspace->beta=1.0/(1e-9 + (wrkspace->tc)[0] );
	printf(" workspace beta set to %f ( 1.0/%f  , %f) \n",wrkspace->beta,wrkspace->tc[0],(wrkspace->tc)[0]);

}

__device__ void updateTrackToVertexProbablilities(Workspace * wrkspace)
{
	if(threadIdx.x==0)
	printf("In the updateTrackToVertexProbablilities\n");

//      >>>>>>>>> KERNELs for  kernel_p_ik <<<<<<<<<
	auto N=wrkspace->nTracks;
	auto CurrentNvetex=wrkspace->nVertex;
	printf("with N = %d , CurrentNvetex = %d",N,CurrentNvetex);
	kernel_p_ik_num<<<CurrentNvetex, N>>>(wrkspace->pik,wrkspace->zt ,wrkspace->zVtx, wrkspace->dz2, wrkspace->beta, N, CurrentNvetex);   	 
	sumBlock_with_loop <<<1,N>>> (wrkspace->pik,wrkspace->pik_denom,\\
					CurrentNvetex,N);
	kernel_p_ik<<<CurrentNvetex, N>>>(wrkspace->pik,wrkspace->pik_denom,N,CurrentNvetex);   	
	cudaDeviceSynchronize();
}

__device__ void updateVertexPositions(Workspace *wrkspace)
{	
	auto N=wrkspace->nTracks;
	auto CurrentNvetex=wrkspace->nVertex;

	if(threadIdx.x==0)
	printf("In the updateVertexPositions\n");
	//      >>>>>>>>>KERNELs for ZVtx Update<<<<<<<<<  
	kernel_z_ik_num<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_numer, wrkspace->pi,wrkspace->zt,wrkspace->dz2, N, CurrentNvetex);
	cudaDeviceSynchronize();
	kernel_z_ik_den<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_denom, wrkspace->pi, wrkspace->zt, wrkspace->dz2, N, CurrentNvetex); 
	cudaDeviceSynchronize();
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_numer, wrkspace->zk_numer, N); 
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_denom, wrkspace->zk_denom, N);  	
	cudaDeviceSynchronize();
	kernel_z_ik(wrkspace->zk_numer, wrkspace->zk_denom,wrkspace->zk_delta ,wrkspace->zVtx, N, CurrentNvetex);  



}

__device__ void updateVertexWeights()
{
	if(threadIdx.x==0)
	printf("In the updateVertexWeights\n");
}

__device__  void updateClusterCriticalTemperatures(Workspace *wrkspace)
{
	auto N=wrkspace->nTracks;
	auto CurrentNvetex=wrkspace->nVertex;

	if(threadIdx.x==0)
	printf("In the updateClusterCriticalTemperatures\n");

	//      >>>>>>>>>KERNEL for T finding <<<<<<<<<	
	kernel_T0_num<<<1, N>>>(wrkspace->tc_numer,wrkspace->zt,\\
					wrkspace->zVtx,wrkspace->pi ,wrkspace->dz2,\\
					N,CurrentNvetex);
	cudaDeviceSynchronize();
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->tc_numer,wrkspace->tc_numer,N);
	// note that the denominator for Zk and Tc_k are same				
	cudaDeviceSynchronize();
	kernel_tc_k(wrkspace->tc_numer,wrkspace->zk_denom,wrkspace->tc,N,CurrentNvetex);
	cudaDeviceSynchronize();
	wrkspace->beta=1.0/(1e-9 + (wrkspace->tc)[0] );

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

	while(i<5 /*workspace.beta < workspace.betaSplitMax */)
	{
   // this could be avoided if we could store a sequnce of betas in the worspace precomputed
		if(threadIdx.x==0)
		  {
		  	workspace.beta*=workspace.betaFactor;
			i+=1;
			printf("at dynamicSplittingPhase with i = %d  zt[0] = %f \n ",i,workspace.zt[0]);
		  }
		 else i++;
		
		auto N=wrkspace->nTracks;
		auto CurrentNvetex=wrkspace->nVertex;
		
		for(int j=0;j<20;j++)
		{

			updateTrackToVertexProbablilities(wrkspace);
		        cudaDeviceSynchronize(); 
			__syncthreads();
			
			updateVertexPositions(wrkspace);
		        cudaDeviceSynchronize(); 
			__syncthreads();
	
			printf("\n\nAt i = %d , j =%d , beta =%f \n",i,j,wrkspace->beta);
			kernel_findFreeEnergyPartA<<<CurrentNvetex,N>>>(wrkspace->FEnergyA,\\
						  wrkspace->zt,wrkspace->zVtx,\\
						   wrkspace->dz2, wrkspace->beta,CurrentNvetex,N);
			kernel_findFreeEnergyPartB<<<1,1>>>(wrkspace->FEnergyA,wrkspace->beta,CurrentNvetex,N);
		
		}
		//updateVertexWeights(wrkspace);
		//__syncthreads();
	
		//updateClusterCriticalTemperatures(wrkspace);
		//__syncthreads();
		//checkAndSplitClusters();
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
		updateTrackToVertexProbablilities(wrkspace);
		__syncthreads();
		updateVertexPositions(wrkspace);
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
	 //udaDeviceSynchronize(); 
	 
	 auto numberOfThreads = 128;
	 auto numberOfBlocks  = (MAXTRACKS/numberOfThreads) + 1;
	
	 loadTracks<<<numberOfBlocks,numberOfThreads>>>(tracks,wrkspace);
	 cudaDeviceSynchronize(); 
	 
	 
	 initializeWorspace<<<256,1024>>>(wrkspace);
	 initializeDAvertexReco<<<1,1>>>(wrkspace);
         cudaDeviceSynchronize(); 
	 printf("\n");
	

	 dynamicSplittingPhase<<<1,1>>>(wrkspace);
	 cudaDeviceSynchronize(); 
	 printf("\n");
	 
	 return nullptr;
	 
	 vertexAssignmentPhase<<<1,102>>>(wrkspace);
	 printf("\n");
	 
	 //printf(cudaGetErrorName(cudaGetLastError()));
	 cudaDeviceSynchronize(); 
	 printf("\n");
	
	 return nullptr;

}

}

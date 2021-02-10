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
__device__ void sumBlock_with_shfl_down_gid_DF(float *in, float *out, int Ntracks,int Nvertices)
{ 

	if(threadIdx.x<Nvertices)
	{
		int vtxId=threadIdx.x*Ntracks;
		int gid = blockIdx.x * blockDim.x + threadIdx.x; 
		for (int offset =1 ; offset <Ntracks; offset ++) // (blockSize/2)+1 
		{        	 
	  		out[vtxId] += in[vtxId+offset] ;     
		}
	}

 
}
__global__ void sumBlock_with_shfl_down_gid_DF_DK(float *in, float *out, int Ntracks,int Nvertices)
{

sumBlock_with_shfl_down_gid_DF(in,out,  Ntracks,Nvertices);

}
__global__ void sumBlock_with_shfl_down_gid(float *in, float *out, int blockSize)
{ 

	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	{        	 
	  out[gid] +=  __shfl_down_sync(0xffffffff, out[gid], offset);     
	}

 
}

__device__ void sumBlock_with_loop_DF(float *in, float *out,int numVertices, int Ntracks)
{ 
	int tid = threadIdx.x; 
	if(tid<Ntracks)
	{
		int off ;
		out[tid]=0.0;	
		for (int offset = 0 ; offset<numVertices; offset++ )  
		{        	  
		  off = Ntracks*offset ;
                  out[tid] += in[tid+off];     
		}
	}
//	printf("out[%d] = %f \n",tid,out[tid]);
}
__global__ void sumBlock_with_loop_DF_DK(float *in, float *out,int numVertices, int Ntracks)
{
    sumBlock_with_loop_DF(in,out,numVertices,Ntracks);
}
__global__ void sumBlock_with_loop(float *in, float *out,int numVertices, int Ntrks)
{ 
	int tid = threadIdx.x; 

	int off ;
	out[tid]=0.0;	
	for (int offset = 0 ; offset<numVertices; offset++ )  
	{        	  
	  off = Ntrks*offset ;
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
//    printf("gid = %d , dz = %f - %f = %f , FEnergyA[gid] = %f\n",gid,zi[idx],zVtx[bid],zi[idx]-zVtx[bid], FEnergyA[gid]);
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
//	printf("( %d , %f ,%f )",i,asum,beta);	
		fEnergy-=logf(asum >1e-20 ? asum : 1.0 )/beta;
	
	}
	printf("\n$(beta , Free Energy) : ,%f,%f\n",beta,fEnergy);
}



__device__ void kernel_p_ik_num_DEVICEFUNC( float *p_ik, float *z_i, float *z_k0,   float *sig, float beta, int Ntracks, int numberOfvertex,int strideMax )
{
	
	auto strideLen = blockDim.x;

	for(auto tid=threadIdx.x;tid<Ntracks;tid+=strideLen)
	{
	    for(auto vid=0;vid<numberOfvertex;vid++)
	    {
              auto gid = vid*Ntracks + tid ;
	      p_ik[gid] =  expf( -beta*(((z_i[tid]-z_k0[vid])*(z_i[tid]-z_k0[vid]))/(sig[tid]*sig[tid]*sig[tid]*sig[tid])) );
        //      printf("DEVICE : gid = %d , dz = %f - %f = %f , pik = %f\n",gid,z_i[tid],z_k0[vid],z_i[tid]-z_k0[vid], p_ik[gid]);
	      //auto x =  expf( -beta*(((z_i[tid]-z_k0[vid])*(z_i[tid]-z_k0[vid]))/(sig[tid]*sig[tid]*sig[tid]*sig[tid])) );
              //printf("DEVICE : gid = %d , dz = %f - %f = %f , pik = %f\n",gid,z_i[tid],z_k0[vid],z_i[tid]-z_k0[vid], x);
	    }
	}
}

__global__ void kernel_p_ik_num_DEVICEFUNC_DUMMY_KERNEL( float *p_ik, float *z_i, float *z_k0,   float *sig, float beta, int Ntracks, int numberOfvertex,int strideMax )
{
    kernel_p_ik_num_DEVICEFUNC( p_ik,z_i,z_k0,sig,beta,Ntracks,numberOfvertex,strideMax);

}
__global__ void kernel_p_ik_num( float *p_ik, float *z_i, float *z_k0,   float *sig, float beta, int N, int numberOfvertex)
{


    int idx = threadIdx.x; 
    int bid = blockIdx.x; // block Id
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    

    if (gid < TotalSize) {
  //     p_ik[gid] =  expf( -beta*(((z_i[idx]-z_k0[bid])*(z_i[idx]-z_k0[bid]))/(sig[idx]*sig[idx]*sig[idx]*sig[idx])) );
    //   printf("GLOBAL : gid = %d , dz = %f - %f = %f , pik = %f\n",gid,z_i[idx],z_k0[bid],z_i[idx]-z_k0[bid], p_ik[gid]);
    }
}

__device__ void kernel_p_ik_DEVICEFUNC( float *p_ik, float *p_ik_den, int Ntracks, int numberOfvertex )
{
    
   auto strideLength=blockDim.x;

   for(auto tid=threadIdx.x;tid<Ntracks;tid+=strideLength)
   {
     if(tid>Ntracks) break;

     for(auto vid=0;vid<numberOfvertex;vid++)
       	{
		auto gid=tid+vid*Ntracks;
		auto oldval = p_ik[gid];
		auto x = -1.0;
		if (p_ik_den[tid] > 1.e-45) 
   	 	{   
   	     		p_ik[gid] =  p_ik[gid]/p_ik_den[tid] ;
   	     		//x =  p_ik[gid]/p_ik_den[tid] ;
   	 	}
   	 	else
   	 	{
	   	     	p_ik[gid] =  0.000 ;     
	   	     	x =  0.000 ;     
   	 	}
//	 printf("DIVICE : pik[%d] = pik_[%d] / p_ik_den[%d] = %f/ %f = %f\n",\\
    		gid,gid,tid,oldval,p_ik_den[tid],p_ik[gid]);
 	
    	}
   
   }
}

__global__ void kernel_p_ik_DEVICEFUNC_DUMMY_KERNEL( float *p_ik, float *p_ik_den, int N, int numberOfvertex )
{

	kernel_p_ik_DEVICEFUNC(p_ik,p_ik_den,N,numberOfvertex);

}
__global__ void kernel_p_ik( float *p_ik, float *p_ik_den, int N, int numberOfvertex )
{
    
    int idx = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; 

    if (gid < TotalSize) 
    { 

	 auto oldval=p_ik[gid];
	 auto x=-1.0;
   	 if (p_ik_den[idx] > 1.e-45) 
   	 {   
   	    // p_ik[gid] =  p_ik[gid]/p_ik_den[idx] ;
   	     x =  p_ik[gid]/p_ik_den[idx] ;
   	 }
   	 else
   	 {
   	     //p_ik[gid] =  0.000 ;     
   	     x =  0.000 ;     
   	 }

 //   printf("GLOBAL : pik[%d] = pik_[%d] / p_ik_den[%d] = %f/ %f = %f\n",\\
    		gid,gid,idx,oldval,p_ik_den[idx],x);
    		//gid,gid,idx,oldval,p_ik_den[idx],p_ik[gid]);
    }

}

__device__ void kernel_z_ik_num_DF( float *p_ik, float *z_ik_num, float *p_i, float *z_i, float *sig, int Ntracks, int numberOfvertex )
{

   auto strideLength=blockDim.x;
   for(auto tid=threadIdx.x;tid<Ntracks;tid+=strideLength)
   {
     if(tid>Ntracks) break;

     for(auto vid=0;vid<numberOfvertex;vid++)
       	{
		auto gid=tid+vid*Ntracks;
                z_ik_num[gid] = p_i[tid]*p_ik[gid]*z_i[tid]/(sig[tid]*sig[tid]); 
	//	printf("DEVICE : z_ik_num[%d] = %f ,tid  = %d , z_i[tid]*p_i[tid]*p_ik[gid]/sig[tid]  = %f*%f*%f/%f^2 ,	sig[tid] = %f \n",gid,z_ik_num[gid],tid,z_i[tid],p_i[tid],p_ik[gid],sig[tid]);
	
	}
   }

}


__global__ void kernel_z_ik_num_DF_DK( float *p_ik, float *z_ik_num, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{

  kernel_z_ik_num_DF( p_ik , z_ik_num , p_i , z_i , sig ,  N ,  numberOfvertex );
  __syncthreads();
}
__global__ void kernel_z_ik_num( float *p_ik, float *z_ik_num, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
    int idx = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
    if (gid < TotalSize) { 
        auto x  = p_i[idx]*p_ik[gid]*z_i[idx]/(sig[idx]*sig[idx]); 
       // printf("GLOBAL : z_ik_num[%d] = %f ,idx  = %d , z_i[idx] = %f , p_i[idx]  = %f ,	sig[idx] = %f \n",gid,x,idx,z_i[idx],p_i[idx],sig[idx]);
       //z_ik_num[gid] = p_i[idx]*p_ik[gid]*z_i[idx]/(sig[idx]*sig[idx]); 
      //  printf("z_ik_num[%d] = %f ,idx  = %d , z_i[idx] = %f , p_i[idx]  = %f ,	sig[idx] = %f \n",gid,z_ik_num[gid],idx,z_i[idx],p_i[idx],sig[idx]);
    }
}

__device__ void kernel_z_ik_den_DF( float *p_ik, float *z_ik_den, float *p_i, float *z_i, float *sig, int Ntracks, int numberOfvertex )
{

   auto strideLength=blockDim.x;
   for(auto tid=threadIdx.x;tid<Ntracks;tid+=strideLength)
   {
     if(tid>Ntracks) break;

     for(auto vid=0;vid<numberOfvertex;vid++)
       	{
		auto gid=tid+vid*Ntracks;
       	z_ik_den[gid] = p_i[tid]*p_ik[gid]/(sig[tid]*sig[tid]); 
       // printf("DEVICE : z_ik_den[%d] = %f ,tid  = %d , p_i[tid]*p_ik[gid]/sig[tid]^2 = %f*%f/%f \n",gid,z_ik_den[gid],tid,p_i[tid],p_ik[gid],sig[tid]);
       //	auto x = p_i[tid]*p_ik[gid]/(sig[tid]*sig[tid]); 
	//     printf("DEVICEL : z_ik_den[%d] = %f ,idx  = %d , p_i[idx]*p_ik[gid]/sig[idx]^2 = %f*%f/%f \n",gid,x,idx,p_i[idx],p_ik[gid],sig[idx]);
 
	}

   }
}
__global__ void kernel_z_ik_den_DF_DK( float *p_ik, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
kernel_z_ik_den_DF( p_ik, z_ik_den, p_i, z_i, sig,N, numberOfvertex );

}

__global__ void kernel_z_ik_den( float *p_ik, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
    int idx = threadIdx.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
    if (gid < TotalSize) {  
        //z_ik_den[gid] = p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
	//printf("GLOBAL : z_ik_den[%d] = %f ,idx  = %d , p_i[idx]  = %f ,	sig[idx] = %f \n",gid,z_ik_den[gid],idx,p_i[idx],sig[idx]);
          auto x        = p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
	//printf("GLOBAL : z_ik_den[%d] = %f ,idx  = %d , p_i[idx]*p_ik[gid]/sig[idx]^2 = %f*%f/%f \n",gid,x,idx,p_i[idx],p_ik[gid],sig[idx]);
    }

}



__device__ void  kernel_z_ik_DF(float * zk_numer,float * zk_denom,float * zDelta,float* zVtx,int ntraks,int currVtxCount )
{
	if(threadIdx.x < currVtxCount)
	{
	  auto newZ=zk_numer[threadIdx.x*ntraks]/(1e-20 + zk_denom[threadIdx.x*ntraks]);
	  zDelta[threadIdx.x] = abs(zVtx[threadIdx.x] - newZ);
	  zVtx[threadIdx.x]   = newZ;
	  printf("DEVICE : setting Z[%d] = %f ,delta[%d] = %f ,numer = %f , deno = %f \n",\\
			threadIdx.x,zVtx[threadIdx.x],threadIdx.x,zDelta[threadIdx.x], zk_numer[threadIdx.x*ntraks],zk_denom[threadIdx.x]);
	}

}

__global__ void  kernel_z_ik_DF_DK(float * zk_numer,float * zk_denom,float * zDelta,float* zVtx,int ntraks,int currVtxCount )
{
	kernel_z_ik_DF( zk_numer, zk_denom, zDelta,zVtx,ntraks,currVtxCount );
}

__global__ void  kernel_z_ik(float * zk_numer,float * zk_denom,float * zDelta,float* zVtx,int ntraks,int currVtxCount )
{
	if(threadIdx.x < currVtxCount)
	{
	  auto newZ=zk_numer[threadIdx.x*ntraks]/(1e-20 + zk_denom[threadIdx.x*ntraks]);
	  auto dz = abs(zVtx[threadIdx.x] - newZ);
	  auto z   = newZ;
	  //zDelta[threadIdx.x] = abs(zVtx[threadIdx.x] - newZ);
	  //zVtx[threadIdx.x]   = newZ;
	  printf("GLOBAL : setting Z[%d] = %f ,delta[%d] = %f ,numer = %f , deno = %f \n",\\
			threadIdx.x,z,threadIdx.x,dz, zk_numer[threadIdx.x*ntraks],zk_denom[threadIdx.x]);
	}
}



__device__ void kernel_T0_num_DF( float *T_num, float *z_i, float *zVtx, float *p,float *p_ik, float *sig, int Ntracks,int numberOfvertex)
{
	auto strideLength=blockDim.x;
	for(auto tid=threadIdx.x;tid<Ntracks;tid+=strideLength)
   	{
     	     if(tid>Ntracks) break;
	     for(auto vid=0;vid<numberOfvertex;vid++)
       	     {
		auto gid=tid+vid*Ntracks;
       		T_num[gid] = p[tid]*p_ik[gid]*((z_i[tid]-zVtx[vid])*(z_i[tid]-zVtx[vid]))/(sig[tid]*sig[tid]); 
 //	      printf("DEVICE : tid = %d, p[tid] =%f , z_i[tid] = %f ,zVtx[%d] =%f ,sig[tid] =%f , Tnum[%d] = %f \n",tid,p[tid],z_i[tid],vid,zVtx[vid],sig[tid],gid,T_num[gid]);

	     }
	}
}
__global__ void kernel_T0_num_DF_DK( float *T_num, float *z_i, float *zVtx, float *p,float *p_ik, float *sig, int N,int currVtxCount )
{

 kernel_T0_num_DF( T_num, z_i, zVtx, p,p_ik, sig, N, currVtxCount );
}
__global__ void kernel_T0_num( float *T_num, float *z_i, float *zVtx, float *p,float *p_ik, float *sig, int N,int currVtxCount )
{
	auto tid = threadIdx.x; 
	auto idx = threadIdx.x; 
	for(auto i=0;i<currVtxCount;i++)
	{
		idx+=N*i;
       		//T_num[idx] = p[tid]*p_ik[idx]*((z_i[tid]-zVtx[i])*(z_i[tid]-zVtx[i]))/(sig[tid]*sig[tid]); 
       		auto x = p[tid]*p_ik[idx]*((z_i[tid]-zVtx[i])*(z_i[tid]-zVtx[i]))/(sig[tid]*sig[tid]); 
 //	printf("GLOBAL : tid = %d, p[tid] =%f , z_i[tid] = %f ,zVtx[%d] =%f ,sig[tid] =%f , Tnum[%d] = %f \n",tid,p[tid],z_i[tid],i,zVtx[i],sig[tid],idx,x);
	}
}

__device__ void  kernel_tc_k_DF(float * tc_numer,float * tc_denom,float* tc,int ntraks,int currVtxCount )
{
	if(threadIdx.x < currVtxCount)
	{
	  tc[threadIdx.x] = 2.0*tc_numer[threadIdx.x*ntraks]/(1e-20 + tc_denom[threadIdx.x*ntraks]);
	 // printf("DEVICE setting tc[%d] = %f , numer = %f , deno = %f \n",\\
			threadIdx.x,tc[threadIdx.x], tc_numer[threadIdx.x*ntraks],tc_denom[threadIdx.x]);
	}
}
__global__ void  kernel_tc_k_DF_DK(float * tc_numer,float * tc_denom,float* tc,int ntraks,int currVtxCount )
{

 kernel_tc_k_DF( tc_numer, tc_denom, tc, ntraks, currVtxCount );

}
__global__ void  kernel_tc_k(float * tc_numer,float * tc_denom,float* tc,int ntraks,int currVtxCount )
{
	if(threadIdx.x < currVtxCount)
	{
	  auto x = 2.0*tc_numer[threadIdx.x*ntraks]/(1e-20 + tc_denom[threadIdx.x*ntraks]);
	  // printf("GLOBAL setting tc[%d] = %f , numer = %f , deno = %f \n",\\
			threadIdx.x,x, tc_numer[threadIdx.x*ntraks],tc_denom[threadIdx.x]);
	}
}

__device__ void check_ifThermalized(float * deltas,float deltaTol ,int *hasThermalized,int currVtxCount)
{
	if(threadIdx.x<currVtxCount)
	{
		if(deltas[threadIdx.x]>deltaTol)
		{
			atomicOr(hasThermalized,1);
		}
	}

}

// probably pass on the z2 avg and spit approximating the xluster to be 2 gaussians
__device__ void kernel_z_k_spliting(float temp,float *z_k, float * tc_clusters ,uint32_t *cur_NV) 
{
/*  
   This kernel take the vertex list and split the last vertex into z-delta,z+delta (delta between 0 and 1.0)
*/

   auto tid= threadIdx.x;
   if (tid >= *cur_NV)
   	return;
  printf("\n\n%d , %d  \n\n",tid,*cur_NV);

   if(temp>tc_clusters[tid])
   {
    printf("Checking for vertex %d at T= %f  and Tc = %f \n ",tid,temp,tc_clusters[tid]);
	return;
   }

   auto idx =  atomicAdd(cur_NV,1);
	
   float z_k_aux =z_k [tid];

   // calculate the deltaZk 
   /*

	auto deltaZk = sqrt( <Z^2>_k - (z_k)^2 )
   */

   auto deltaZk  = abs(0.2*z_k[tid]);
   z_k[tid] = z_k_aux - deltaZk;
   z_k[idx] = z_k_aux + deltaZk;

   printf("Checking for vertex %d at T= %f  and Tc = %f, delta = %f z_old = %f z_new[%d] = %f\n ",tid,temp,tc_clusters[tid],deltaZk,z_k[tid],idx,z_k[idx]);
}
__global__ void kernel_z_k_spliting_DF_DK(float temp,float *z_k, float * tc_clusters ,uint32_t *cur_NV) 
{

   kernel_z_k_spliting(temp,z_k, tc_clusters ,cur_NV) ;
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
	printf("at initialization N  = %d \n",N);
	auto CurrentNvetex = 1;
	auto numThreads=1024;
	//      >>>>>>>>>KERNELs for ZVtx Update<<<<<<<<<  
	kernel_z_ik_num<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_numer, wrkspace->pi,wrkspace->zt,wrkspace->dz2, N, CurrentNvetex);
	cudaDeviceSynchronize();
	kernel_z_ik_num_DF_DK<<<1, numThreads>>>(wrkspace->pik, wrkspace->zk_numer, wrkspace->pi,wrkspace->zt,wrkspace->dz2, N, CurrentNvetex);
	cudaDeviceSynchronize();
	kernel_z_ik_den<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_denom, wrkspace->pi, wrkspace->zt, wrkspace->dz2, N, CurrentNvetex); 
	cudaDeviceSynchronize();
	kernel_z_ik_den_DF_DK<<<1, numThreads>>>(wrkspace->pik, wrkspace->zk_denom, wrkspace->pi, wrkspace->zt, wrkspace->dz2, N, CurrentNvetex); 
	cudaDeviceSynchronize();
	//sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_numer, wrkspace->zk_numer, N); 
	sumBlock_with_shfl_down_gid_DF_DK<<<1,numThreads>>>(wrkspace->zk_numer, wrkspace->zk_numer, N,CurrentNvetex); 
	//sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_denom, wrkspace->zk_denom, N);  	
	sumBlock_with_shfl_down_gid_DF_DK<<<1, numThreads>>>(wrkspace->zk_denom, wrkspace->zk_denom, N,CurrentNvetex);  	
	cudaDeviceSynchronize();
	kernel_z_ik<<<1,CurrentNvetex>>>(wrkspace->zk_numer, wrkspace->zk_denom,wrkspace->zk_delta ,wrkspace->zVtx, N, CurrentNvetex);  
	kernel_z_ik_DF_DK<<<1,numThreads>>>(wrkspace->zk_numer, wrkspace->zk_denom,wrkspace->zk_delta ,wrkspace->zVtx, N, CurrentNvetex);  

	wrkspace->nVertex=1;
 	 	
	//      >>>>>>>>>KERNEL for T finding <<<<<<<<<	
	kernel_T0_num<<<1, N>>>(wrkspace->tc_numer,wrkspace->zt,\\
					wrkspace->zVtx,wrkspace->pi,wrkspace->pik ,wrkspace->dz2,\\
					N,CurrentNvetex);
	kernel_T0_num_DF_DK<<<1,numThreads>>>(wrkspace->tc_numer,wrkspace->zt,\\
					wrkspace->zVtx,wrkspace->pi,wrkspace->pik ,wrkspace->dz2,\\
					N,CurrentNvetex);

	cudaDeviceSynchronize();
	//sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->tc_numer,wrkspace->tc_numer,N);
	sumBlock_with_shfl_down_gid_DF_DK<<<1,numThreads>>>(wrkspace->tc_numer,wrkspace->tc_numer,N, CurrentNvetex);
	// note that the denominator for Zk and Tc_k are same				
	cudaDeviceSynchronize();
	kernel_tc_k<<<1,CurrentNvetex>>>(wrkspace->tc_numer,wrkspace->zk_denom,wrkspace->tc,N,CurrentNvetex);
	kernel_tc_k_DF_DK<<<1,numThreads>>>(wrkspace->tc_numer,wrkspace->zk_denom,wrkspace->tc,N,CurrentNvetex);
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
	auto numberOfThreads =1024;
	printf("with N = %d , CurrentNvetex = %d \n",N,CurrentNvetex);
	
	kernel_p_ik_num<<<CurrentNvetex, N>>>(wrkspace->pik,wrkspace->zt ,wrkspace->zVtx, wrkspace->dz2, wrkspace->beta, N, CurrentNvetex);   	 
	
	cudaDeviceSynchronize();
	auto numThreads=1024;
	auto stride_max=( int((numThreads-1)/N) + 1 );
	kernel_p_ik_num_DEVICEFUNC_DUMMY_KERNEL<<<1,numThreads>>>(wrkspace->pik,wrkspace->zt ,wrkspace->zVtx, wrkspace->dz2, wrkspace->beta, N, CurrentNvetex, stride_max);   	 
	
	//sumBlock_with_loop <<<1,N>>> (wrkspace->pik,wrkspace->pik_denom,CurrentNvetex,N);
	sumBlock_with_loop_DF_DK<<<1,numberOfThreads>>>(wrkspace->pik,wrkspace->pik_denom,CurrentNvetex,N);
	cudaDeviceSynchronize();
	__syncthreads();
	kernel_p_ik<<<CurrentNvetex, N>>>(wrkspace->pik,wrkspace->pik_denom,N,CurrentNvetex);   	
	kernel_p_ik_DEVICEFUNC_DUMMY_KERNEL<<<1,numThreads>>>(wrkspace->pik,wrkspace->pik_denom,N,CurrentNvetex);   	
	cudaDeviceSynchronize();
}

__device__ void updateVertexPositions(Workspace *wrkspace)
{	
	auto N=wrkspace->nTracks;
	auto CurrentNvetex=wrkspace->nVertex;
	auto numThreads=1024;

	if(threadIdx.x==0)
	printf("In the updateVertexPositions wit %d vertexes \n",wrkspace->nVertex);
	//      >>>>>>>>>KERNELs for ZVtx Update<<<<<<<<<  
	kernel_z_ik_num<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_numer, wrkspace->pi,wrkspace->zt,wrkspace->dz2, N, CurrentNvetex);
	cudaDeviceSynchronize();
	kernel_z_ik_num_DF_DK<<<1, numThreads>>>(wrkspace->pik, wrkspace->zk_numer, wrkspace->pi,wrkspace->zt,wrkspace->dz2, N, CurrentNvetex);
	cudaDeviceSynchronize();
	kernel_z_ik_den<<<CurrentNvetex, N>>>(wrkspace->pik, wrkspace->zk_denom, wrkspace->pi, wrkspace->zt, wrkspace->dz2, N, CurrentNvetex); 
	cudaDeviceSynchronize();
	kernel_z_ik_den_DF_DK<<<1, numThreads>>>(wrkspace->pik, wrkspace->zk_denom, wrkspace->pi, wrkspace->zt, wrkspace->dz2, N, CurrentNvetex); 
	cudaDeviceSynchronize();
	//sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_numer, wrkspace->zk_numer, N); 
	sumBlock_with_shfl_down_gid_DF_DK<<<1,numThreads>>>(wrkspace->zk_numer, wrkspace->zk_numer, N,CurrentNvetex); 
	cudaDeviceSynchronize();
	//sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->zk_denom, wrkspace->zk_denom, N);  	
	sumBlock_with_shfl_down_gid_DF_DK<<<1,numThreads>>>(wrkspace->zk_denom, wrkspace->zk_denom, N,CurrentNvetex);  	
	cudaDeviceSynchronize();
	kernel_z_ik<<<1,CurrentNvetex>>>(wrkspace->zk_numer, wrkspace->zk_denom,wrkspace->zk_delta ,wrkspace->zVtx, N, CurrentNvetex);  
	kernel_z_ik_DF_DK<<<1,numThreads>>>(wrkspace->zk_numer, wrkspace->zk_denom,wrkspace->zk_delta ,wrkspace->zVtx, N, CurrentNvetex);  



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
	auto numberOfThreads=1024;
	if(threadIdx.x==0)
	printf("In the updateClusterCriticalTemperatures\n");

	//      >>>>>>>>>KERNEL for T finding <<<<<<<<<	
	kernel_T0_num<<<1, N>>>(wrkspace->tc_numer,wrkspace->zt,\\
					wrkspace->zVtx,wrkspace->pi, wrkspace->pik ,wrkspace->dz2,\\
					N,CurrentNvetex);
	kernel_T0_num_DF_DK<<<1, numberOfThreads>>>(wrkspace->tc_numer,wrkspace->zt,\\
					wrkspace->zVtx,wrkspace->pi, wrkspace->pik ,wrkspace->dz2,\\
					N,CurrentNvetex);
	
	cudaDeviceSynchronize();
	//sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(wrkspace->tc_numer,wrkspace->tc_numer,N);
	sumBlock_with_shfl_down_gid_DF_DK<<<1,numberOfThreads>>>(wrkspace->tc_numer,wrkspace->tc_numer,N,CurrentNvetex);
	// note that the denominator for Zk and Tc_k are same				
	cudaDeviceSynchronize();
	kernel_tc_k<<<1,CurrentNvetex>>>(wrkspace->tc_numer,wrkspace->zk_denom,wrkspace->tc,N,CurrentNvetex);
	kernel_tc_k_DF_DK<<<1,numberOfThreads>>>(wrkspace->tc_numer,wrkspace->zk_denom,wrkspace->tc,N,CurrentNvetex);
	cudaDeviceSynchronize();

}

__device__ void checkAndSplitClusters(Workspace *wrkspace)
{
	if(threadIdx.x==0)
	printf("In the checkAndSplitClusters\n");
	auto CurrentNvetex = wrkspace->nVertex;
	auto numberOfThreads=1024;
	//kernel_z_k_spliting<<<1,CurrentNvetex>>>(1.0/wrkspace->beta,wrkspace->zVtx,wrkspace->tc,&(wrkspace->nVertex) );
	kernel_z_k_spliting_DF_DK<<<1,numberOfThreads>>>(1.0/wrkspace->beta,wrkspace->zVtx,wrkspace->tc,&(wrkspace->nVertex) );
	cudaDeviceSynchronize();
	printf("Numver of vertices after checkAndSplitClusters = %d \n",wrkspace->nVertex);

}

__device__ void checkAndMergeClusters(Workspace *wrkspace)
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
			printf("at dynamicSplittingPhase with i = %d  zt[0] = %f  , zVtx[0] = %f , beta = %f \n ",i,workspace.zt[0],workspace.zVtx[0],workspace.beta);
		  }
		 else i++;

		__syncthreads();
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

			//check_ifThermalized<<<1,1>>>(wrkspace->zk_delta,0.001,wrkspace->hasThermalized,CurrentNvetex);
			check_ifThermalized(wrkspace->zk_delta,0.001,wrkspace->hasThermalized,CurrentNvetex);
			
			cudaDeviceSynchronize();

			if(*(wrkspace->hasThermalized)==0)
			{
		printf("has thermalized for beta = %f , j =%d\n",wrkspace->beta,j);
				break;
			}

			*(wrkspace->hasThermalized)=0;
		}
		//updateVertexWeights(wrkspace);
		//__syncthreads();
	
	        updateClusterCriticalTemperatures(wrkspace);
		cudaDeviceSynchronize();
		__syncthreads();
		checkAndSplitClusters(wrkspace);
		cudaDeviceSynchronize();
		
		for(int ii=0;ii<wrkspace->nVertex;ii++)
		   printf("vertex [%d] = %f \n",ii,wrkspace->zVtx[ii]);
		}

	//checkAndMergeClusters();
	for(int ii=0;ii<wrkspace->nVertex;ii++)
		   printf("*vertex [%d], %f \n",ii,wrkspace->zVtx[ii]);
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
		//checkAndMergeClusters();
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

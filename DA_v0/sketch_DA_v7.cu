#include  <cassert>
#include <iostream> 
#include <fstream>
using namespace std;

#include <curand.h>
#include <curand_kernel.h>


__global__ void kernel( float *a, int N )
{
//int idx = blockIdx.x*blockDim.x + threadIdx.x;
int idx = threadIdx.x;
//a[idx] = 7;
    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       a[idx] = 7;
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );


}
 
__global__ void kernel1( float *a, int N )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x; 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       a[idx] = 9;
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );

    if (idx < N) {
    printf("idx  %d input[idx] %f \n", idx, a[idx] );
       a[idx] += idx;
    }
}


__global__ void kernel2( float *a, int N )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x; 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       a[idx] = 9;
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );


}

__global__ void kernel3( float *num, float *p, int N )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x; 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       num[idx] = p[idx]+12;
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );


}

__global__ void kernel4( float *num, float *z_i, float *p, float *sig, int N )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x; 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       num[idx] = p[idx]*z_i[idx]/(sig[idx]*sig[idx]);
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );


}

__global__ void z_0_numdem( float *num, float *den, float *z_i, float *p, float *sig, int N )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x; 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       num[idx] = p[idx]*z_i[idx]/(sig[idx]*sig[idx]);
       den[idx] = p[idx]/(sig[idx]*sig[idx]);
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );


}
 

__global__ void kernel_T_0( float *T_num, float *z_i, float z_k0, float *p, float *sig, int N )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x; 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       T_num[idx] = p[idx]*((z_i[idx]-z_k0)*(z_i[idx]-z_k0))/(sig[idx]*sig[idx]*sig[idx]*sig[idx]); 
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );


}


//__global__ void kernel_p_ik_num( float *p_ik, float *z_i, float *z_k0, float *rho, float *sig, int beta, int N, int numberOfvertex )
__global__ void kernel_p_ik_num( float *p_ik, float *z_i, float *z_k0,   float *sig, float beta, int N, int numberOfvertex )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
    int idx = threadIdx.x; 
    int bid = blockIdx.x; // block Id
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    

/*
    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       p_ik[idx] = p[idx]*((z_i[idx]-z_k0)*(z_i[idx]-z_k0))/(sig[idx]*sig[idx]*sig[idx]*sig[idx]); 
    }
*/
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );
    if (gid < TotalSize) {
       p_ik[gid] =  expf(-beta*(((z_i[idx]-z_k0[bid])*(z_i[idx]-z_k0[bid]))/(sig[idx]*sig[idx]*sig[idx]*sig[idx])) );
       
       
      //  p_ik[gid] =  exp(-beta*(z_i[idx]-z_k0[bid])*(z_i[idx]-z_k0[bid]));
        
        
        // expf(-beta*(((z_i[idx]-z_k0[bid])*(z_i[idx]-z_k0[bid]))/(sig[idx]*sig[idx]*sig[idx]*sig[idx])) );
        //printf("idx %d bid %d gid %d z_i[idx] %f z_k0[bid] %f p_ik[gid] %.10e   beta_d %f\n", idx, bid, gid, z_i[idx],z_k0[bid], p_ik[gid], beta  );
    }

}



__global__ void kernel_p_ik_den( float *p_ik, float *p_ik_den, float *rho, int N)
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
    // kernel must be
    // kernel_p_ik_den <<<blocks,sub_list>>>   
    // blocks = number of clusters , sub_list = data size
*/
    int idx = threadIdx.x;  
    int gid = blockIdx.x * blockDim.x + idx; 
    //int oid = blockIdx.x * blockDim.x; 
    //int bid = blockIdx.x;
    
    p_ik_den[gid] = rho[idx]*p_ik[gid];
    
    for (int offset =  __float2uint_ru(float(N)/2)  ; offset > 0; offset /= 2)  	 
    {        	 
       p_ik_den[gid] +=  __shfl_down_sync(0xffffffff, p_ik_den[gid], offset);       
       //printf(" offset  %d , idx  %d , bid %d oid %d , gid  %d , p_ik_den[gid] %e,  p_ik[gid] %e \n", offset, idx, bid, oid, gid, p_ik_den[gid],  p_ik[gid] );  //%.10e
    }     
}


__global__ void kernel_p_ik( float *p_ik, float *p_ik_den, int N, int numberOfvertex )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
    int idx = threadIdx.x; 
    //int bid = blockIdx.x; // block Id
    int oid = blockIdx.x * blockDim.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    

/*
    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       p_ik[idx] = p[idx]*((z_i[idx]-z_k0)*(z_i[idx]-z_k0))/(sig[idx]*sig[idx]*sig[idx]*sig[idx]); 
    }
*/
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );
    if (gid < TotalSize) { 
    
    if (p_ik_den[oid] > 1.e-45) {   
        p_ik[gid] =  p_ik[gid]/p_ik_den[oid] ;
        }
        else{  
        p_ik[gid] =  0.000 ;     
        
        }//  
        //printf("idx %d gid %d z_i[idx] %f z_k0[bid] %f p_ik[gid] %.10e \n", idx, gid, z_i[idx],z_k0[bid], p_ik[gid] );
        //printf("ok ........... ");
    }

}




__global__ void kernel_z_ik_numden( float *p_ik, float *z_ik_num, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
    int idx = threadIdx.x; 
    //int bid = blockIdx.x; // block Id
    //int oid = blockIdx.x * blockDim.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
 
 
    if (gid < TotalSize) { 
        z_ik_num[gid] = p_i[idx]*p_ik[gid]*z_i[idx]/(sig[idx]*sig[idx]);
        z_ik_den[gid] = p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
        
    //printf("idx %d gid %d oid %d z_ik_num[gid]  %.10e z_ik_den[gid]  %.10e \n", idx, gid, oid, z_ik_num[gid],z_ik_den[gid] );
        //printf("ok ........... ");
    }

}


__global__ void kernel_z_ik_num( float *p_ik, float *z_ik_num, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
    int idx = threadIdx.x; 
    //int bid = blockIdx.x; // block Id
    //int oid = blockIdx.x * blockDim.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
 
 
    if (gid < TotalSize) { 
        z_ik_num[gid] = p_i[idx]*p_ik[gid]*z_i[idx]/(sig[idx]*sig[idx]); 
        
    //printf("idx %d gid %d oid %d z_ik_num[gid]  %.10e \n", idx, gid, oid, z_ik_num[gid] );
        //printf("ok ........... ");
    }

}


__global__ void kernel_z_ik_den( float *p_ik, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
    int idx = threadIdx.x; 
    //int bid = blockIdx.x; // block Id
    //int oid = blockIdx.x * blockDim.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex    
 
 
    if (gid < TotalSize) {  
        z_ik_den[gid] = p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
        
    //printf("idx %d gid %d oid %d  z_ik_den[gid]  %.10e \n", idx, gid, oid, z_ik_den[gid] );
        //printf("ok ........... ");
    }

}

__global__ void sumBlock_with_loop(float *in, float *out, int blockSize)
{ 
	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	int tid = threadIdx.x; 
	//int bid = blockIdx.x ;
	int off ;
	//  !!!!!!  Warning Warp Divergence !!!!!! 
	out[tid] = in[tid]; 
	//printf(" __float2uint_ru(float(blockSize)/2) %d gridDim.x %d ",__float2uint_ru(float(blockSize)/2),gridDim.x)
 
	for (int offset =  __float2uint_ru(float(gridDim.x)/2)  ; offset > 0; offset /= 2)  
	{        	  
	  //off = blockSize*  offset *gridDim.x + tid; 
	  off = blockSize* offset + tid; 
	  
	  out[gid] =  out[gid] + in[off];     
	 //printf(" off %d tid %d  bid %d gid %d off  %d out[tid] %.5e  out[gid] %.5e  in[gid] %.5e   in[off] %.5e \n",offset, tid, bid, gid, off, out[tid] , out[gid], in[gid], in[off]);    
	}  
 
} 


/*
__global__ void kernel_z_ik( float *z_ik_num, float *z_ik_den, int N, int numberOfvertex )
{
// p = track weigth;  z_i = tracks;
//   sig = deviation;    N =  N of tracks

    int idx = threadIdx.x; 
    //int bid = blockIdx.x; // block Id
    int ofid = blockIdx.x * blockDim.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex     
 
    if (gid < TotalSize) {  
        z_ik_num[gid] = z_ik_num[gid]/z_ik_den[ofid]; //  p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
        
   // printf("=>depois idx %d gid %d ofid %d  z_ik_num[gid]  %.10e  z_ik_den[gid]  %.10e \n", idx, gid, ofid, z_ik_num[gid], z_ik_den[gid] );
        //printf("ok ........... ");
    }

}
*/


__global__ void kernel_z_ik(float *z_k, float *z_ik_num, float *z_ik_den, int N, int numberOfvertex )
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
    int idx = threadIdx.x; 
    //int bid = blockIdx.x; // block Id
    int ofid = blockIdx.x * blockDim.x; 
    int gid = blockIdx.x * blockDim.x + idx;
    int TotalSize = N*numberOfvertex; //  nTracks * nVertex     
 
    if (gid < TotalSize) {  
        z_k[gid] = z_ik_num[gid]/z_ik_den[ofid]; //  p_i[idx]*p_ik[gid]/(sig[idx]*sig[idx]); 
        
    //printf("=>depois idx %d gid %d bid %d ofid %d  z_ik_num[gid]  %.10e  z_ik_den[gid]  %.10e  z_k[gid]  %.10e \n", idx, gid, bid, ofid, z_ik_num[gid], z_ik_den[gid],z_k[gid] );
        //printf("ok ........... ");
    }

}

/*
// This is just a example to show the allocation inside kernel is not good
__global__ void kernel2( float *a, const int N )
{
// p = track weigth;  z_i = tracks;
// sig = deviation;    N =  N of tracks
//
int idx = threadIdx.x; 
//float test0_d[N], test1_d[N]; // Test if a list inside kernel works
long test0_d[N];   // not work only   long d[1000]; // https://stackoverflow.com/questions/2187189/creating-arrays-in-nvidia-cuda-kernel 

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       test0_d[idx] = 9;
    }
    //printf("idx  %d output[idx] %f \n", idx, a[idx] );

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       test1_d[idx] = idx;
    }
    
a = test0_d;    
}

*/


__global__ void kernel_z_0_dontknow(float *num,float *dem, float *p, float *z_i, float *sig ,int N ) 
{
/* p = track weigth;  z_i = tracks;
   sig = deviation;    N =  N of tracks
*/
int idx = threadIdx.x;  

    if (idx < N) {
    //printf("idx  %d input[idx] %f \n", idx, a[idx] );
       num[idx] = p[idx]*z_i[idx]/sig[idx];
       dem[idx] = p[idx]/sig[idx];
    }

    
}

__global__ void sumBlock_with_shfl_down(float *in, float *out, int blockSize)
{ 
	//int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	int tid = threadIdx.x; 
	//  !!!!!!  Warning Warp Divergence !!!!!! 
	out[tid] = in[tid];
	//printf(" gid %d in[gid]=%f, out[gid]=%f  \n",gid, in[gid], out[gid]);
	//printf(" blockSize %d ",blockSize);	
	//printf(" gid %d ",gid);
	//printf(" tid %d ",tid);
 
	for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	// do not work for (int offset =  1.0*blockSize/2  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	
	// To round up A[idx] = __float2uint_rz( tmp1 ) ;
	{        	 
	  //printf(" offset %d ",offset);
	   
	  out[tid] +=  __shfl_down_sync(0xffffffff, out[tid], offset);     
	  //printf(" off %d tid %d out[tid] %f \n",offset, tid, out[tid] );    
	}  
 
} 


__global__ void sumBlock_with_shfl_down_gid(float *in, float *out, int blockSize)
{ 
	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	//int tid = threadIdx.x; 
	//int bid = blockIdx.x ;
	//int aux = __float2uint_ru(float(blockSize)/2) ;
	//  !!!!!!  Warning Warp Divergence !!!!!! 
	//out[gid] = in[gid];
	//printf(" gid %d in[gid]=%f, out[gid]=%f  \n",gid, in[gid], out[gid]);
	//printf(" blockSize %d ",blockSize);	
	//printf(" gid %d ",gid);
	//printf(" tid %d ",tid);
 
	//for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; --offset ) // (blockSize/2)+1  
	//lst for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	// do not work for (int offset =  1.0*blockSize/2  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	
	// To round up A[idx] = __float2uint_rz( tmp1 ) ;
	{        	 
	  //printf(" offset %d ",offset);
	  //aux = __float2uint_ru(float(offset)/2)  ; 
	  out[gid] +=  __shfl_down_sync(0xffffffff, out[gid], offset);     
	  //printf(" off %d tid %d  bid %d gid %d  out[tid] %f out[gid] %f \n",offset, tid, bid, gid, out[tid] , out[gid]);  
	  //printf(" aux %d  \n",aux );  
	    
	}  
 
}  


__global__ void sumBlock_with_shfl_down_k(float *in, float *out, int blockSize)
{ 
	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	//int oid = blockIdx.x * blockDim.x ;	
	//int tid = threadIdx.x; 
	//int bid = blockIdx.x ;
	//int aux = __float2uint_ru(float(blockSize)/2) ;
	//  !!!!!!  Warning Warp Divergence !!!!!! 
	out[gid] = in[gid];
	//printf(" gid %d in[gid]=%f, out[gid]=%f  \n",gid, in[gid], out[gid]);
	//printf(" blockSize %d ",blockSize);	
	//printf(" gid %d ",gid);
	//printf(" tid %d ",tid);
 
	//for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; --offset ) // (blockSize/2)+1  
	//lst for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	for (int offset =  __float2uint_ru(float(blockSize)/2)  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	// do not work for (int offset =  1.0*blockSize/2  ; offset > 0; offset /= 2) // (blockSize/2)+1 
	
	// To round up A[idx] = __float2uint_rz( tmp1 ) ;
	{        	 
	  //printf(" offset %d ",offset);
	  //aux = __float2uint_ru(float(offset)/2)  ; 
	  out[gid] +=  __shfl_down_sync(0xffffffff, out[gid], offset);     
	  //printf(" off %d tid %d  bid %d oid %d gid %d  out[tid] %f out[gid] %f \n",offset, tid, bid, oid, gid, out[tid] , out[gid]);  
	  //printf(" aux %d  \n",aux );  
	    
	}  
 
} 



//****************************************************//
//***  Kernels for =====> RANDOM NUMBERS <==== ******//
//****************************************************//

//http://ianfinlayson.net/class/cpsc425/notes/cuda-random
// It is important to includethe below libs in the head of this file
//#include <curand.h>
//#include <curand_kernel.h> 

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, float* numbers2) {
  /* curand works like rand - except that it takes a state as a parameter */ 
  numbers2[blockIdx.x]= curand_uniform(&states[blockIdx.x]) ;
   
}


// Function to generate random numbers
float* CUDA_uniform_rand_list(int N, float * numbers2_h) {
  // N= Number of values (size of output array)
  // numbers2_h= Alloc list that will receive the random numbers generated by the device
  
  curandState_t* states;   
  float *numbers2_d;       // Device list
  int nBytes; nBytes = N*sizeof(float);  
  
  
  numbers2_h = (float *)malloc(nBytes);    // Allocate Host list

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, N * sizeof(curandState_t));        // Allocate Device list
  cudaMalloc((void**) &numbers2_d, N * sizeof(curandState_t));    // Allocate Device list

  /* invoke the GPU to initialize all of the random states */
  init<<<N, 1>>>(time(0), states); 

  /* invoke the kernel to get some random numbers */
  randoms<<<N, 1>>>(states, numbers2_d);

  /* copy the random numbers back */ 
  cudaMemcpy(numbers2_h, numbers2_d, nBytes, cudaMemcpyDeviceToHost); 

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);


    return numbers2_h;
}

// End kernels for random numbers



//********************************************************//
//*** Kernels for =====> kernel_z_k_spliting <==== ******//
//******************************************************//
__device__ int cur_NV_d=0; // Must be N-1, because for first vertex corresponts to element 0 of the list.

__global__ void kernel_z_k_spliting(float *z_k, float *DeltRand, int cur_NV) 
{
/*  
   This kernel take the vertex list and split the last vertex into z-delta,z+delta (delta between 0 and 1.0)

   z_k = full vertex list;    
   cur_NV = CurrentNvetex current number of vertex;    
   it takes __device__ float cur_NV_d=0 as well;
*/

   float z_k_aux =z_k [cur_NV];
   //printf("device cur_NV %d\n",cur_NV);
   //printf("device z_k_aux %f\n",z_k_aux);
   printf("-device stating device cur_NV %d cur_NV_d, z_k [0] %f z_k [cur_NV] %f z_k [cur_NV+1] %f\n", cur_NV, cur_NV_d, z_k[0] , z_k [cur_NV],z_k [cur_NV+1] );
//cur_NV = cur_NV_d;

   z_k [cur_NV]   = z_k[cur_NV] - DeltRand[cur_NV];      // z-delta
   z_k [cur_NV+1] = z_k_aux + DeltRand[cur_NV];  

//printf("device cur_NV %d cur_NV_d,z_k [0] %f z_k [cur_NV] %f z_k [cur_NV+1] %f\n", z_k[0] , cur_NV_d,z_k [cur_NV],z_k [cur_NV+1] );

   //printf("-device finishing device cur_NV %d cur_NV_d, z_k [0] %f z_k [cur_NV] %f z_k [cur_NV+1] %f\n", cur_NV, cur_NV_d, z_k[0] , z_k [cur_NV],z_k [cur_NV+1] );
   printf("-device finishing, cur_NV %d cur_NV_d, z_k[0] %f, z_k[1] %f , z_k[2] %f \n", cur_NV, cur_NV_d, z_k[0] , z_k[1],z_k [2] );
   cur_NV_d = cur_NV+1;   // Incrementing number of avaiable vertex. Defined in __device__ float cur_NV_d=0; 
   //printf("device cur_NV_d after %d\n",cur_NV_d);

}


int main(void)
{
	//float *a_h, *b_h, *c_h; // host data
	float z_0, T_0;  // host only data
	float *tracks_h, *z_h, *sig_h, *p_h, *rho_h, *num_h, *den_h; // host data 
	float  *T_num_h; // host data 
	float beta = 1.0e-3;
	
	//float *a_d, *b_d; // device data
	float *num_d, *den_d; // device data
	float *tracks_d, *z_d, *sig_d, *p_d, *rho_d; // device data
	float *T_num_d; // device data
	
	int N = 15, nBytes_data, nBytes_Vertex, i ;
	int MaxNVetex	= 3 ;
	
	// AUXILIARY variables
	int CurrentNvetex = 0; 
	
	float Toydata[N]= {1.0,2.0,3.0,4.0,5.0,15.0,16.0,17.0,18.0,19.0,47.0 , 48.0 , 49.0 , 50.0 , 51.0};
	
	
	/*
	for (i=0; i< N; i++)
	{	 
	 
	 std::cout << "| Toydata["<<i << "]=" << Toydata[i];  
	 
	 }
	 std::cout << " end 0 "<< std::endl; 
	 std::cout << "   "<< std::endl;  
	*/
	
	
	nBytes_data = N*sizeof(float); 
	
	
	tracks_h = (float *)malloc(nBytes_data);
	z_h = (float *)malloc(nBytes_data);
	sig_h = (float *)malloc(nBytes_data);
	p_h = (float *)malloc(nBytes_data);
	rho_h = (float *)malloc(nBytes_data);
	num_h = (float *)malloc(nBytes_data);
	den_h = (float *)malloc(nBytes_data);
	T_num_h = (float *)malloc(nBytes_data);
	
	 
	
	//cudaMalloc((void **) &a_d, nBytes);
	//cudaMalloc((void **) &b_d, nBytes);
	
	
	cudaMalloc((void **) &num_d, nBytes_data);	
	cudaMalloc((void **) &den_d, nBytes_data);
	cudaMalloc((void **) &tracks_d, nBytes_data);	

	cudaMalloc((void **) &sig_d, nBytes_data);	
	cudaMalloc((void **) &p_d, nBytes_data);
	cudaMalloc((void **) &rho_d, nBytes_data);
	cudaMalloc((void **) &T_num_d, nBytes_data);
	
	
	nBytes_Vertex = MaxNVetex*sizeof(float); 	
	
	cudaMalloc((void **) &z_d, nBytes_Vertex);
	
	
	// Host Variables Initialization
	for (i=0; i<N; i++){
	 
	 num_h[i] = 0.0;
	 den_h[i] = 0.0;
	 tracks_h[i] = Toydata[i];	 
	 sig_h[i] = 1.0;
	 p_h[i] = 1.0;
	 rho_h[i] = 1.0;
	  
	 }
	 //std::cout << " end 1 "<< std::endl;  
	 
	  
	 
	 //   CudaMemcpyHostToDevice
        cudaMemcpy(num_d, num_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(den_d, den_h, nBytes_data, cudaMemcpyHostToDevice);        
        cudaMemcpy(tracks_d, tracks_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(sig_d, sig_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(p_d, p_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(rho_d, rho_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(T_num_d, T_num_h, nBytes_data, cudaMemcpyHostToDevice);
        
        cudaMemcpy(z_d, z_h, nBytes_Vertex, cudaMemcpyHostToDevice);
	

	//   Start process 
	
 	std::cout << "-pre-z_0_numdem Calculating numerator and denominator for z_0"<< std::endl;  
 	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	//      >>>>>>>>>KERNEL<<<<<<<<<	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	
	
        // NUMERATOR and DENOMINATOR z_0
	z_0_numdem<<<1, N>>>(num_d, den_d, tracks_d, p_d, sig_d, N);      // Calculating numerator and denominator for z_0
	
 
	cudaMemcpy(num_h, num_d, nBytes_data, cudaMemcpyDeviceToHost);  
	cudaMemcpy(den_h, den_d, nBytes_data, cudaMemcpyDeviceToHost); 	
	
	std::cout << "-pos-kernel z_0_numdem "<< std::endl;  	
	
	// To print some results for Debug	num_h, den_h
/*
		for (i=0; i< N; i++)
	{	 
	 std::cout << "| num_h["<<i << "]=" << num_h[i];  
	 
	 }
	 std::cout << " end 2 "<< std::endl;  
	 std::cout << "   "<< std::endl;  	 	
		for (i=0; i< N; i++)
	{	 
	 std::cout << "| den_h["<<i << "]=" << den_h[i];  
	 
	 }
	 std::cout << " end 3 "<< std::endl;  
	 std::cout << "   "<< std::endl;  
*/		 	
	 	
	 	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	//      >>>>>>>>>KERNEL<<<<<<<<<	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
		 	
		 	
        // NUMERATOR sum and DENOMINATOR sum z_0
	sumBlock_with_shfl_down<<<1, N>>>(num_d, num_d, N);       // Calculating numerator sum for z_0
	sumBlock_with_shfl_down<<<1, N>>>(den_d, den_d, N);       // Calculating denominator sum for z_0
	
	cudaMemcpy(den_h, den_d, nBytes_data, cudaMemcpyDeviceToHost); 
	cudaMemcpy(num_h, num_d, nBytes_data, cudaMemcpyDeviceToHost);  
	
	z_0 = num_h[0]/den_h[0];
	z_h[0] = z_0;
	cudaMemcpy(z_d, z_h, nBytes_Vertex, cudaMemcpyHostToDevice);
	
	std::cout << "  z_0 = "<< z_0 <<std::endl ; 	
	std::cout << "  z_h[0] = "<< z_h[0]  <<std::endl ; 	
	
	std::cout << "-pos-kernel sumBlock_with_shfl_down for z_0 "<< std::endl;  	
		
		
		
	std::cout << "Create random list z_0_seed (mean) "<< z_0 <<std::endl ; 	
	
	  float * randDelta_h;  
	  //int nBytes1; nBytes1 = MaxNVetex*sizeof(float);  
	  randDelta_h = (float *)malloc(nBytes_Vertex);    // Allocate Host list 
	  
 	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	//      >>>>>>>>>KERNEL<<<<<<<<<	      RANDOM
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
		  
	  
        // Creating a list with random uniform numbers 0 to 1
	  randDelta_h = CUDA_uniform_rand_list( MaxNVetex, randDelta_h);  // Do not forget to   free(numbers_h);
	  

/*
	// To print some results for Debug	randDelta_h, num_h, den_h
	 for (i=0; i< 5; i++) {std::cout << "| randDelta_h["<<i << "]=" << randDelta_h[i]; }
	 std::cout << " end 4 "<< std::endl;  
	 std::cout << "   "    << std::endl;  		   
 
	std::cout << " pos-function CUDA_uniform_rand_list for random delta => z_0+delta "<< std::endl;  	
  
	 for (i=0; i< N; i++) { std::cout << "| num_h["<<i << "]=" << num_h[i]; }
	 std::cout << " end 5 "<< std::endl;  
	 std::cout << "   "<< std::endl;  		
  
	 for (i=0; i< N; i++) { std::cout << "| den_h["<<i << "]=" << den_h[i]; }
	 std::cout << " end 6 "<< std::endl;  
	 std::cout << "   "<< std::endl; 
*/ 	

  
 	 	
 	 	
 	 	
 	 	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	//      >>>>>>>>>KERNEL<<<<<<<<<	 kernel_T_0
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	
	std::cout << "-pre-kernel kernel_T_0 for T_0" << std::endl; 
	kernel_T_0<<<1, N>>>(T_num_d, tracks_d, z_0,p_d, sig_d, N);     // Calculating numerator for T_0
	std::cout << " pre-kernel sumBlock_with_shfl_down for T_0" << std::endl; 
	sumBlock_with_shfl_down<<<1, N>>>(T_num_d, T_num_d, N);        // Calculating numerator sum for T_0
		
	cudaMemcpy(T_num_h, T_num_d, nBytes_data, cudaMemcpyDeviceToHost);  
	//std::cout << " aqui den_h[0]  " << den_h[0] << std::endl;   
	
	T_0 = 2.*2.*T_num_h[0]/den_h[0]; //The second 2.* is related with the contribution of the second vertex z+delta z-delta (which is similar)
			 
	std::cout << "-pos-kernel T_0=> T_0 = "<< T_0 << std::endl; 
  
	// To print some results for Debug	T_num_h	    
/*
		for (i=0; i< N; i++) {	 
	 //std::cout << "| T_num_h["<<i << "]=" << T_num_h[i];  
	 std::cout <<  T_num_h[i] << ","; 	 
	 }
	 std::cout << " end 7 "<< std::endl;  
	 std::cout << "   "<< std::endl; 
*/
	 
	 
	 
 
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	//      >>>>>>>>>KERNEL<<<<<<<<<	   kernel_z_k_spliting
	//      >>>>>>>>>>>><<<<<<<<<<<< 
		 
	//__global__ void kernel_z_k_spliting(float *z_k, float *DeltRand, int cur_NV) 	v
	
	float * randDelta_d;  
	cudaMalloc((void **) &randDelta_d, nBytes_Vertex);  
	cudaMemcpy(randDelta_d, randDelta_h, nBytes_Vertex, cudaMemcpyHostToDevice);  
	 
	std::cout << "   "<< std::endl; 	
	std::cout << "   "<< std::endl; 	
	
	cudaMemcpy(z_d,z_h, nBytes_Vertex, cudaMemcpyHostToDevice);	 
	//__global__ void kernel_z_k_spliting(float *z_k, float *DeltRand, int cur_NV) 	  
	kernel_z_k_spliting<<<1, 1>>>(z_d, randDelta_d, CurrentNvetex);   
	cudaMemcpy(z_h, z_d, nBytes_Vertex, cudaMemcpyDeviceToHost);	 
	std::cout << "-pos-kernel 1 kernel_z_k_spliting= "<< z_h[0] <<" | "<< z_h[1] <<" | "<< z_h[2] <<std::endl; 
	cudaMemcpy(z_d,z_h, nBytes_Vertex, cudaMemcpyHostToDevice);
		
	CurrentNvetex = 1; 	 	  
	kernel_z_k_spliting<<<1, 1>>>(z_d, randDelta_d, CurrentNvetex);  
	cudaMemcpy(z_h, z_d, nBytes_Vertex, cudaMemcpyDeviceToHost);	 	
	 	
	std::cout << "-pos-kernel 2 kernel_z_k_spliting= "<< z_h[0] <<" | "<< z_h[1] <<" | "<< z_h[2] <<std::endl; 
	//cudaMemcpy(z_d, z_h, nBytes_Vertex, cudaMemcpyHostToDevice);	 
	//cudaMemcpy(z_h, z_d, nBytes_Vertex, cudaMemcpyDeviceToHost);	 
	 CurrentNvetex = 2; 
    
		for (i=0; i< N; i++)
	{	 
	 //std::cout << "| T_num_h["<<i << "]=" << T_num_h[i];  
	 std::cout << "z_h["<< i<< "]= " << z_h[i] << ",  "; 
	 
	 }
	 std::cout << " end 8 "<< std::endl;  
	 std::cout << "   "<< std::endl; 	 	 
	 
 	
 	float *p_ik_h, *p_ik_d;
 	float *p_ik_den_h, *p_ik_den_d; 	
 	
 	int Total = N*MaxNVetex;
 	int nBytes_p_ik; 
	
	nBytes_p_ik = Total*sizeof(float);  	//for (i=0; i<Total; i++){p_ik_h[i]= ;}
	
	 	
	p_ik_h = (float *)malloc(nBytes_p_ik); 
	p_ik_den_h = (float *)malloc(nBytes_p_ik); 	

	cudaMalloc((void **) &p_ik_d, nBytes_p_ik);	
	cudaMalloc((void **) &p_ik_den_d, nBytes_p_ik);	
	
	
	
	float *z_ik_num_h, *z_ik_num_d; // 	 
	float *z_ik_den_h, *z_ik_den_d; // 
	
	z_ik_num_h = (float *)malloc(nBytes_p_ik); 
	z_ik_den_h = (float *)malloc(nBytes_p_ik); 	
	
 	//for (i=0; i< Total; i++)	{z_ik_den_h [i] = 0.0; } 
 
	cudaMalloc((void **) &z_ik_num_d, nBytes_p_ik);	
	
	
	 
	 
	 
	 
	 
	 
	 
	 
	 
	 
	
	// DO A LOOP

  int len_beta = 5 ;	
  float beta_range [len_beta] = {1.0e-03, 5.0e-03, 9.0e-03, 1.0e-02, 2.5e-02};
  //  int len_beta =21;	
  //float beta_range [len_beta] = {1.0e-03, 5.0e-03, 9.0e-03, 1.0e-02, 2.5e-02, 4.0e-02, 5.5e-02,7.0e-02, 8.5e-02, 1.0e-01, 1.0e-01, 6.0e-01, 1.1e+00, 1.6e+00, 2.1e+00, 2.6e+00, 3.1e+00, 3.6e+00, 4.1e+00, 4.6e+00, 5.1e+00};
  
  
  
  
  std::cout << "-z_k before loop = "<< z_h[0] <<" | "<< z_h[1] <<" | "<< z_h[2] <<std::endl; 
	
  for (int i_b=0; i_b<2; i_b++){ 	 //len_beta       
	beta = beta_range[i_b];
	printf("===== %d to %d beta %f  =====\n",i_b,len_beta-1,beta );
	std::cout << "-z_k beta loop = "<< z_h[0] <<" | "<< z_h[1] <<" | "<< z_h[2] <<std::endl; 
	std::cout << "    "  << std::endl; 
		
        cudaMemcpy(p_ik_d, p_ik_h, nBytes_p_ik, cudaMemcpyHostToDevice);	 	
	  	
 	 	
	//      >>>>>>>>>>>><<<<<<<<<<<< 	
	//      >>>>>>>>>KERNEL<<<<<<<<<    kernel_p_ik_num	 kernel_p_ik_den   kernel_p_ik
	//      >>>>>>>>>>>><<<<<<<<<<<< 
	
	CurrentNvetex = 3; 	 
	//  void kernel_p_ik( float *p_ik, float *z_i, float *z_k0, float *rho, float *sig, int beta, int N, int numberOfvertex )
	 std::cout << "-pre kernel kernel_p_ik_num \n"<< std::endl;	
	kernel_p_ik_num<<<CurrentNvetex, N>>>(p_ik_d, tracks_d, z_d, sig_d, beta, N, CurrentNvetex);   	 
	
	cudaMemcpy(p_ik_h, p_ik_d, nBytes_p_ik, cudaMemcpyDeviceToHost);	 	



	// C3PO
	 
	// std::cout << "-pos kernel kernel_p_ik_num \n"<< std::endl;
	 //std::cout << " p_ik_h \n"<< std::endl;  
//	for (i=0; i< Total; i++)	{ printf("after i  %d  p_ik_h[i] %.10e \n",i, p_ik_h[i]); }
	 //for (i=0; i< Total; i++)	{ printf(" %.10e, ", p_ik_h[i]); }
	 //std::cout << " end 9 "<< std::endl;  
//	 std::cout << "       "<< std::endl; 
 
	 
	//int nBytes1; nBytes1 = MaxNVetex*sizeof(float);  
	 
	 
	//__global__ void kernel_p_ik_den( float *p_ik, float *p_ik_den, float *rho, int N)	 
	//       <<<currentNvetex,Ntracks>>>
	
 
	 for (i=0; i< Total; i++)	{  p_ik_den_h[i]= p_ik_h[i]; }	
	 /*
	 for (i=0; i< Total; i++)	{ 	 
	     if (i%15 == 0) { 
        printf("\n");
        }
	 printf(" p_ik_den_h %.10e, ", p_ik_den_h[i]); 
	 }
	 std::cout << " end 10 "<< std::endl;  
	 */
	 
        cudaMemcpy(p_ik_den_d, p_ik_den_h, nBytes_p_ik, cudaMemcpyHostToDevice);	
        	
	 //kernel_p_ik_den <<<3,Total>>> (p_ik_d, p_ik_den_d, rho_d, Total); 
	 printf("-pre-kernel sumBlock_with_loop \n");
	 sumBlock_with_loop <<<3,N>>> (p_ik_den_d,p_ik_den_d,N);
	 printf("-pos-kernel sumBlock_with_loop \n");
	 	 
	 cudaMemcpy(p_ik_den_h, p_ik_den_d, nBytes_p_ik, cudaMemcpyDeviceToHost);	 
	 //std::cout << "\n=======  p_ik_den_h[i] \n "<< std::endl;  
	 //for (i=0; i< Total; i++)	{ printf("after i  %d  p_ik_den_h[i] %e ",i, p_ik_den_h[i]); } //%.10e 
	 //std::cout << "\n end 11 "<< std::endl;  
	 //std::cout << "       "<< std::endl; 	 
	 
	  
        cudaMemcpy(p_ik_d, p_ik_h, nBytes_p_ik, cudaMemcpyHostToDevice);	 	 
        cudaMemcpy(p_ik_den_d, p_ik_den_h, nBytes_p_ik, cudaMemcpyHostToDevice);		 
	 
	 
	 
//__global__ void kernel_p_ik( float *p_ik, float *p_ik_den, int N, int numberOfvertex )
	printf("-pre-kernel kernel_p_ik \n");
	kernel_p_ik<<<CurrentNvetex, N>>>(p_ik_d, p_ik_den_d, N, CurrentNvetex);   	
	printf("-pos-kernel kernel_p_ik \n");	 
	 
	cudaMemcpy(p_ik_h, p_ik_d, nBytes_p_ik, cudaMemcpyDeviceToHost);		 
	 
 
	 //std::cout << "\n final p_ik_h[i] \n "<< std::endl;  
	 //for (i=0; i< Total; i++)	{ printf("after i  %d  p_ik_h[i] %e \n",i, p_ik_h[i]); } //%.10e 
	 //std::cout << " end 11 "<< std::endl;  
	 //std::cout << "       "<< std::endl; 		
 
	 
	 
	 
	 
	 
	 
  //************************************************// 
 //********************* Here *********************//
 
 
// void kernel_z_ik_numden( float *p_ik, float *z_ik_num, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )	 
	 
		
        cudaMemcpy(z_ik_num_d, z_ik_num_h, nBytes_p_ik, cudaMemcpyHostToDevice);	
        	
	
 // void kernel_z_ik_numden( float *p_ik, float *z_ik_num, float *z_ik_den, float *p_i, float *z_i, float *sig, int N, int numberOfvertex )
 
 	// for (i=0; i< Total; i++)	{ printf("pre i  %d  z_ik_den_h[i] %e \n",i, z_ik_den_h[i]); } //%.10e 
 
	//kernel_z_ik_numden<<<CurrentNvetex, N>>>(p_ik_d, z_ik_num_d, p_ik_den_d, p_d, tracks_d, sig_d, N, CurrentNvetex); 

	printf("-pre-kernel kernel_z_ik_num \n");	 
	kernel_z_ik_num<<<CurrentNvetex, N>>>(p_ik_d, z_ik_num_d, p_d, tracks_d, sig_d, N, CurrentNvetex);
	printf("-pos-kernel kernel_z_ik_num \n");
	cudaMemcpy(z_ik_num_h, z_ik_num_d, nBytes_p_ik, cudaMemcpyDeviceToHost);

        //for (i=0; i< Total; i++)	{ printf("after i  %d  z_ik_num_h[i] %e   \n",i, z_ik_num_h[i] ); } //%.10e 	
 	// std::cout << " end 12 "<< std::endl;  
 	// std::cout << "       "<< std::endl; 	
 

	cudaMalloc((void **) &z_ik_den_d, nBytes_p_ik);			 	
        cudaMemcpy(z_ik_den_d, z_ik_den_h, nBytes_p_ik, cudaMemcpyHostToDevice);    
            
	printf("-pre-kernel kernel_z_ik_den \n");	
	kernel_z_ik_den<<<CurrentNvetex, N>>>(p_ik_d, z_ik_den_d, p_d, tracks_d, sig_d, N, CurrentNvetex); 
	printf("-pos-kernel kernel_z_ik_num \n");
	 		 
	cudaMemcpy(z_ik_den_h, z_ik_den_d, nBytes_p_ik, cudaMemcpyDeviceToHost);	 
	 
   
 /*
        for (i=0; i< Total; i++)	{ printf("after i  %d  z_ik_den_h[i] %e   \n",i, z_ik_den_h[i] ); } //%.10e 	
 	 std::cout << " end 11 "<< std::endl;  
 	 std::cout << "       "<< std::endl; 	
 */
	 
	 	 
	//free(a_h); free(b_h); cudaFree(a_d); cudaFree(b_d);
	
	cudaMalloc((void **) &z_ik_num_d, nBytes_p_ik);		
        cudaMemcpy(z_ik_num_d, z_ik_num_h, nBytes_p_ik, cudaMemcpyHostToDevice);		
	
// void sumBlock_with_shfl_down_gid(float *in, float *out, int blockSize)
//__global__ void sumBlock_with_shfl_down_gid(float *in, float *out, int blockSize)	
	
	printf("-pre-kernel sumBlock_with_shfl_down_gid \n");		
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(z_ik_num_d, z_ik_num_d, N); 
	printf("-pos-kernel sumBlock_with_shfl_down_gid \n");
	//sumBlock_with_loop <<<blocks,sub_list>>> (z_ik_num_d,z_ik_num_d,sub_list); 
	
		
	
 	 //std::cout << " ====== "<< std::endl;  
 	 //std::cout << "       "<< std::endl; 	
	
	cudaMemcpy(z_ik_num_h, z_ik_num_d, nBytes_p_ik, cudaMemcpyDeviceToHost);	
	
	
        //for (i=0; i< Total; i++)	{ printf("after sum  %d  z_ik_num_h[i] %e   \n",i, z_ik_num_h[i] ); } //%.10e 	
 	// std::cout << " end 13 "<< std::endl;  
 	// std::cout << "        "<< std::endl; 
 	 
 	 
	cudaMemcpy(z_ik_den_d, z_ik_den_h, nBytes_p_ik, cudaMemcpyHostToDevice);    
	
	printf("-pre-kernel sumBlock_with_shfl_down_gid \n");	
	sumBlock_with_shfl_down_gid<<<CurrentNvetex, N>>>(z_ik_den_d, z_ik_den_d, N);  	
	printf("-pos-kernel sumBlock_with_shfl_down_gid \n");	
	 
	cudaMemcpy(z_ik_den_h, z_ik_den_d, nBytes_p_ik, cudaMemcpyDeviceToHost);	  	 
 	 	


        cudaMemcpy(z_ik_num_d, z_ik_num_h, nBytes_p_ik, cudaMemcpyHostToDevice);
        cudaMemcpy(z_ik_den_d, z_ik_den_h, nBytes_p_ik, cudaMemcpyHostToDevice); 
        
        cudaMemcpy(z_d,z_h, nBytes_Vertex, cudaMemcpyHostToDevice);	
        
	std::cout << "-pre-kernel kernel_z_ik= "<< z_h[0] <<" | "<< z_h[1] <<" | "<< z_h[2] <<std::endl;             
/*
         for (i=0; i< Total; i++)	{ printf("aqui  %d  z_ik_num_h[i] %e   \n",i, z_ik_num_h[i] ); } //%.10e 	
 	 std::cout << " end 12 "<< std::endl;  
 	 std::cout << "        "<< std::endl; 
*/

//__global__ void kernel_z_ik( float *z_ik_num, float *z_ik_den, int N, int numberOfvertex ) 	 

 	// std::cout << "\n kernel_z_ik\n "<< std::endl;
	printf("-pre-kernel kernel_z_ik \n");  
	kernel_z_ik<<<CurrentNvetex, N>>>(z_d, z_ik_num_d, z_ik_den_d, N, CurrentNvetex);  
	printf("-pos-kernel kernel_z_ik \n");   
		
	 
	//cudaMemcpy(z_ik_num_h, z_ik_num_d, nBytes_p_ik, cudaMemcpyDeviceToHost);
        cudaMemcpy(z_h, z_d, nBytes_Vertex, cudaMemcpyDeviceToHost);		

	//std::cout << " \n "<< std::endl; 	
	std::cout << "-pos-kernel kernel_z_ik= "<< z_h[0] <<" | "<< z_h[1] <<" | "<< z_h[2] <<std::endl;    
		
 	// std::cout << " \n "<< std::endl;  	
        //for (i=0; i< Total; i++)	{ printf("after sum  %d  z_ik_den_h[i] %e   \n",i, z_ik_num_h[i] ); } //%.10e 	
 	 std::cout << " end 14 "<< std::endl;  
 	 //std::cout << "        "<< std::endl;  
 	 
 	 std::cout << " \n "<< std::endl;  	
        for (i=0; i< MaxNVetex; i++)	{ printf("after i  %d  z_h[i] %e   \n",i, z_h[i] ); } //%.10e 	
 	 std::cout << " end 15 "<< std::endl;  
 	 std::cout << "        "<< std::endl;   	 
 	 
 	 
 	 
 	 
 	 }
 	 
 	 
 	 
 
 	 std::cout << " The End       "<< std::endl; 	 	 	
 	 	
 	 	
	free(tracks_h); free(z_h);  	
	free(sig_h); free(p_h); 	
	free(rho_h); free(num_h); 	
	free(den_h); free(T_num_h); 
	free(randDelta_h); free(p_ik_h); 
	free(p_ik_den_h);  
	
	
	
	cudaFree(tracks_d); cudaFree(sig_d); cudaFree(p_d);	
	cudaFree(rho_d); cudaFree(T_num_d); cudaFree(z_d);
	cudaFree(randDelta_d);    
	cudaFree(p_ik_d); cudaFree(p_ik_den_d);
	//cudaFree(z_ik_num_d); 
	cudaFree(z_ik_den_d);
	
	
	
	//cudaFree(sig_d); cudaFree(p_d);
	
	
	
	return 0;
}

 

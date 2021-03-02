
#include <assert.h>

#include <stdio.h>
#include <iostream>
#include <fstream> 

using namespace std;

__device__ int comp_list_2n_blocks(int list_size) {        
      
      
      return int(pow(2,(__float2uint_ru(log10f(list_size)/log10f(2.)))));
 } 

__global__ void sumBlock_with_shfl_and_par_reduc(float *in, float *aux, float *out, float *summ, int N_warps_per_vertex, int N_tracks, int N_vertex){ 

	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	int tid = threadIdx.x; 
	int bid = blockIdx.x ;
	int warp =32;
	int num = bid%N_warps_per_vertex;
	int div = bid/N_warps_per_vertex; 
	int Did = num * warp + div * N_tracks + tid ;
	int Lid = (N_warps_per_vertex -1 )% N_warps_per_vertex ;  // Last warp of a group of threads
	int Rid = N_tracks%warp ;  // Last warp of a group of threads
	int Sid = (tid + (num * warp) + (div * N_warps_per_vertex )) * warp; // 
	int warp_elem = N_warps_per_vertex * warp * (div+1);      //  variable with maximum element value in each big block
	int N_2W = comp_list_2n_blocks(N_warps_per_vertex);      // Call comp_list_2n_blocks to know to round up to 2**n number of elements for parallel  reduction
	int Pid = num * warp + tid; // variable index to limit the parallel reduction 
 
 
	
	if (num != Lid){
	
	    //printf("if bid %d, Lid %d,num %d,tid %d, Did %d,gid %d,in[Did] %f\n",bid,Lid,num,tid,Did,gid,in[Did]);
	out[gid] = in[Did];    	
	}
	else {
        	if (tid<Rid){   	//  !!!!!!  Warning Warp Divergence !!!!!! 
	
	    //printf("el bid %d, Lid %d,num %d,tid %d, Did %d,gid %d,in[Did] %f\n",bid,Lid,num,tid,Did,gid,in[Did]);	
	 
	out[gid] = in[Did];     	
	                    }
	                       	    
	}
 
	  __syncthreads();              // Wait for all shuffle reductions per loop	
	  
	  
	// ======>  NEXT Step Sum reduction with   __shfl_down_sync
	  
        // Performing shfl reduction summation over warps
        // Results of the summation over each warp will be in the first elements of the warp
	for (int offset =  blockDim.x/2  ; offset > 0; offset /= 2) //  
	 
	{        	  
		  
	  out[gid] +=  __shfl_down_sync(0xffffffff, out[gid], offset);  
	  
	  // Without index [gid] gives error => out+=__shfl_down_sync(0xffffffff, out , offset);  	
              
	  __syncthreads();              // Wait for all shuffle reductions per loop
	  
	} 	



 
	// ======>  NEXT Step distribute result from shfl sum for parallel reduction

 
  __syncthreads();	  
	if (Sid < warp_elem){   
  __syncthreads();	
	//Sid
	aux[gid] = out[Sid]; // Saver option
	
//printf("b %d,t %d,w_e %d,d+1 %d, g %d, Sid %d, out[S] %f\n",bid,tid,warp_elem, (div+1),gid,Sid,out[Sid] );
	}
	  
	  
	// ======>  NEXT Step Parallel reduction	
	
  __syncthreads();	  
	if (Pid < N_2W/2){   
  //__syncthreads();	
  
 for (int offset = N_2W/2 ; offset >0; offset /= 2)  {
 
  aux[gid] += aux[gid+offset];
  __syncthreads();
 //printf("2b %d,t %d, g %d, Sid %d, aux[S] %f, aux[g+o] %f, N2 %d, P %d, o %d\n",bid,tid, gid,Sid,aux[gid],aux[gid+offset],N_2W,Pid,offset);
 }
  
  
	//Sid
	//aux[gid] = out[Sid]; // Saver option
	
//printf("b %d,t %d,w_e %d,d+1 %d, g %d, Sid %d, out[S] %f\n",bid,tid,warp_elem, (div+1),gid,Sid,out[Sid] );
	}	
	
	
	
// if you want print sum results in order
__syncthreads();
if (gid < N_vertex){
__syncthreads();
   summ[gid]= aux[gid*warp*N_warps_per_vertex];
 //printf("2b %d,t %d, g %d, Sid %d, aux[S] %f, aux[g+o] %f, N2 %d, P %d, o %d\n",bid,tid, gid,Sid,aux[gid],aux[gid+offset],N_2W,Pid,offset);
}

else{

   summ[gid]= 0.0;

}
	__syncthreads();
	
	
	
	
 }

 
// Requires #include <math.h> 
 int increase_pow_2 (int list_size) {  
     
     return int(ceil(log10(list_size)/log10(2.)));
 }
 
 
  int warp_fit(int list_size) {  
      
     return int(ceil(list_size/32. ));
 }
 
  int warp_fit_2n(int list_size) {        
      
      int n_warps= int(ceil(list_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(pow(2.,ceil(log10(n_warps)/log10(2.))));
 } 
 
 
  int N_blocks_warp(int list_size,int tracks_size,int N_vertex) {        
      
      int n_warps= int(ceil(tracks_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(N_vertex*n_warps);
 }  
 
 
  int N_sub_blocks_warp(int tracks_size) {        
      
      int n_warps= int(ceil(tracks_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(n_warps);
 }
 

int main() {

  int N = 102;
  int N_tracks = 34;
  float toy_data [N] = {0,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100,101}; 
  int warp = 32, N_vertex= 3;
  int N_blocks,N_threads, N_list_full;
  
  
// Calculate how many warps is necessary to perform the summation
  int data_n_warp_n =  warp_fit(N_tracks);
  int data_n_warp_2n =  warp_fit_2n(N_tracks);
  int block_data_warp =  N_blocks_warp(N_tracks,N_tracks,N_vertex); 
  int N_warps_per_vertex = N_sub_blocks_warp(N_tracks);
  N_blocks = block_data_warp;
  N_threads = warp;
  N_list_full = N_blocks * N_threads;
  
   
  printf("Total number of warps (same as number of blocks) = %d \n",N_blocks); 
  printf("Subblocks number warps (same as number of warps per tracks, or blocks per tracks) = %d \n",N_warps_per_vertex); 
  printf("Total size of the auxiliary list = %d \n",N_list_full); 
  
  
   
  printf("aux_h has this size = %d \n",N_blocks); 
  printf("out_h has this size = %d \n",N);  
  printf("total gid has this size = %d \n",N_blocks*warp); 
  
  
  
  float *teste_h,*out_h,*aux_h,*S_vi_h,*teste_d,*out_d,*aux_d,*S_vi_d;
  
  int nBytes_data = N*sizeof(float); 
  int nBytes_output = N_list_full*sizeof(float); 
  int nBytes_sum = N_vertex*sizeof(float); 
  
  
  teste_h = (float *)malloc(nBytes_data); 
  out_h = (float *)malloc(nBytes_output); 
  aux_h = (float *)malloc(nBytes_output); 
  S_vi_h = (float *)malloc(nBytes_sum); 
  
  
  cudaMalloc((void **) &teste_d, nBytes_data);  
  cudaMalloc((void **) &out_d, nBytes_output); 
  cudaMalloc((void **) &aux_d, nBytes_output); 
  cudaMalloc((void **) &S_vi_d, nBytes_sum); 
  
    
  for (int i=0; i<N; i++){ teste_h[i] = toy_data[i]; }   
  
  
  cudaMemcpy(teste_d, teste_h, nBytes_data, cudaMemcpyHostToDevice);   
  cudaMemcpy(out_d, out_h, nBytes_output, cudaMemcpyHostToDevice);  
  cudaMemcpy(S_vi_d, S_vi_h, nBytes_sum, cudaMemcpyHostToDevice);   
  
  //>>>>>>>>>>>>>>>>>>>>>     KERNEL <<<<<<<<<<<<<<<<<<<<<<<<<<//  
  //>>>>>>>>>>>>>>>>>>>>>     KERNEL <<<<<<<<<<<<<<<<<<<<<<<<<<//
  //>>>>>>>>>>>>>>>>>>>>>     KERNEL <<<<<<<<<<<<<<<<<<<<<<<<<<//  
  
  //__global__ void sumBlock_with_shfl_and_par_reduc(float *in, float *out,  int N_warps_per_vertex)
  sumBlock_with_shfl_and_par_reduc <<<N_blocks,warp>>>  (teste_d,aux_d,out_d,S_vi_d,N_warps_per_vertex,N_tracks,N_vertex);
  

  cudaMemcpy(out_h, out_d, nBytes_output, cudaMemcpyDeviceToHost); 
  cudaMemcpy(teste_h, teste_d, nBytes_output, cudaMemcpyDeviceToHost); 
  cudaMemcpy(aux_h, aux_d, nBytes_output, cudaMemcpyDeviceToHost); 
  cudaMemcpy(S_vi_h, S_vi_d, nBytes_sum, cudaMemcpyDeviceToHost);   
  
    
  for (int i=0; i<N_vertex; i++){   printf("i= %d,  S_vi_h = %f \n",i,S_vi_h[i]);   }   
  printf("The correct answer is 528.0, 1650.0, 2873.0\n");
  
  
  
  
  
  
}  
  

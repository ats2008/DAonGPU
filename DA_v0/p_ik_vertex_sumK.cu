
#include <assert.h>

#include <stdio.h>
#include <iostream>
#include <fstream> 

using namespace std;

//***************************************************//
//** Host fuctions (could be device, think later) **//
//**************************************************//

// Requires #include <math.h> 
 int increase_pow_2 (int list_size) {  
     
     return int(ceil(log10(list_size)/log10(2.)));
 }
 
 int two_to_increase_pow_2(int list_size) {  
     
     return pow(2.,int(ceil(log10(list_size)/log10(2.))) );
 } 
 
  int warp_fit(int list_size) {  
      
     return int(ceil(list_size/32. ));
 }
 
  int warp_fit_2n(int list_size) {        
      
      int n_warps= int(ceil(list_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(pow(2.,ceil(log10(n_warps)/log10(2.))));
 } 
 
 

  
  int N_sub_blocks_warp(int tracks_size) {        
  
      // Number of warp per number of tracks
      
      int n_warps= int(ceil(tracks_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(n_warps);
 }
 
   int N_blocks_warp(int tracks_size,int N_vertex) {        
      
            // Number of warp per number of tracks for all vertex
      
      int n_warps= int(ceil(tracks_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(N_vertex*n_warps);
 }  
 
   int N_T_blocks_warp(int tracks_size,int N_vertex) {      
     
      // Total number of elements to split tracks over warps
      int warp = 32;
      int n_warps= int(ceil(tracks_size/32. ));
      //printf("n_warps %d \n",n_warps);
      return int(N_vertex*n_warps*warp);
 }  
 
 
 
//****************************************************//
//**************** Device functions *****************//
//**************************************************//


__device__ int comp_list_2n_blocks(int list_size) {        
      
      // Calculates the total number of elements for list_size to be proportional to 2**n
      // list_size can be other object that you want to know what is its complement to 2**n
      return int(pow(2,(__float2uint_ru(log10f(list_size)/log10f(2.)))));
 } 
 
 //***********************************************//
 //**********************************************//
 //*********************************************//

__global__ void kernel_p_ik_pre_Block_warp(float *p_ik, float *den_h, float *z_i, float *z_k, float *sig, float *rho,float *aux, float beta, int N_warps_per_vertex, int N_tracks, int N_vertex){ 

	//  p_ik = Boltzmann list (filled at the end) ; z_i = z-coordinates tracks; z_k = z-coordinates vertex position;
       //  sig =  track measurement uncertainty;       aux = auxiliary list to spread data over warps;
      // beta = inverse of the temperature ;          N_warps_per_vertex = N. of warps per vertex;
     // N_tracks = N. of tracks;                     N_vertex = N. of vertex
    // <<<blocks,threads>>> ==>   <<<Total N warps,warps>>> 


	int gid = blockIdx.x * blockDim.x + threadIdx.x;  
	int tid = threadIdx.x;                           
	int bid = blockIdx.x ;
	int warp =32;
	int num = bid%N_warps_per_vertex;               // The index that changes repeatedly through warps inside the big block 
	int div = bid/N_warps_per_vertex;               // The index that changes only through big blocks (and warps)  
	int Did = num * warp  + tid ;   // Index to spread data correctly through. Note that for p_ik we do not use div since we are running only through z_i
	int Lid = (N_warps_per_vertex -1 )% N_warps_per_vertex ;  // Last warp inside the big block 
	int Rid = N_tracks%warp ;  // Required number of threads in the last warp inside a group of threads
	//int Sid = (tid + (num * warp) + (div * N_warps_per_vertex )) * warp; // The index that takes the result from shfl and spread within threads (inside the big block)
	//int warp_elem = N_warps_per_vertex * warp * (div+1);      //  Variable with maximum element value in each big block
	int N_2W = comp_list_2n_blocks(N_warps_per_vertex);      // Call comp_list_2n_blocks to know the round up to 2**n number of elements for parallel  reduction
	//int Pid = num * warp + tid; // Index to limit the parallel reduction 
	//int Vid = N_vertex * N_warps_per_vertex * warp/gid; //  Vertex index, this index goes from zero to N. vertex (constant among warps within sub-blocks)
	//int PiD = Vid * N_warps_per_vertex * warp + num * warp + tid; // Paste each parallel result into a sequential list
	int div_g =  gid/N_tracks; 
	int rem1 = gid%N_tracks ; //the remainder, 0 1 2 ... and repeat (goes back to zero) when arising in N_tracks
	//int Sid = div * N_warps_per_vertex * warp + (div_g * N_warps_per_vertex * warp) + rem1;//  index to shrink data over warps to a sequential form
	int Sid =  (div_g * N_warps_per_vertex * warp) + rem1;//  index to shrink data over warps to a sequential form
 
 // In this realization we try to be economical in memory think, we calculate p_ik numerator with aux, and after all calculation, we use it again for the denominator. Another possibility is to include an extra aux2 to carry the aux, multiply it by rho_k, and rearrange to denominator final list.
	
	if (num != Lid){
	
	aux[gid] =  exp(-beta*(((z_i[Did]-z_k[div])*(z_i[Did]-z_k[div]))/(sig[Did]*sig[Did]*sig[Did]*sig[Did])) );
	//printf("if bid %d, Lid %d,num %d,div %d,tid %d, Did %d,gid %d,z_i[Did] %.4e,z_k[div] %.4e, aux[Did] %.4e\n",bid,Lid,num,div,tid,Did,gid,z_i[Did],z_k[div],aux[gid]);  	
	
	}
	else {
        	if (tid<Rid){   	//  !!!!!!  Warning Warp Divergence !!!!!! 
	 
	aux[gid] =  exp(-beta*(((z_i[Did]-z_k[div])*(z_i[Did]-z_k[div]))/(sig[Did]*sig[Did]*sig[Did]*sig[Did])) );
	//printf("ifelif bid %d, Lid %d,num %d,div %d,tid %d, Did %d,gid %d,z_i[Did] %.4e,z_k[div] %.4e, aux[Did] %.4e\n",bid,Lid,num,div,tid,Did,gid,z_i[Did],z_k[div],aux[gid]); 	  	
	
	                    }
        	else{
	                    aux[gid] = 0.0; 
	                    
	//printf("ifelifel bid %d, Lid %d,num %d,div %d,tid %d, Did %d,gid %d,z_i[Did] %.4e,z_k[div] %.4e, aux[Did] %.4e\n",bid,Lid,num,div,tid,Did,gid,z_i[Did],z_k[div],aux[gid]); 	 
	                    }	                       	    
	}
 
	  __syncthreads();              // Wait for all threads to finish the above operation	
	  
	  
	// ======>  NEXT Step rearrange to exclude artificial zeros 
	
       // if you want print sum results in order 
 
	
	
	if (gid < N_tracks * N_vertex){
	
	p_ik [gid] = aux[Sid] ;
	__syncthreads(); 
	//printf("b %d, L %d,num %d,div %d,t %d, D %d,g %d, Sid %d,div_g %d, rem1 %d, aux[S] %.3e ,p_ik [g] %.3e\n",bid,Lid,num,div,tid,Did,gid,Sid,div_g,rem1,aux[Sid],p_ik [gid]);  	
	
	}
  
	 //  *****************************************************  //  
	//  ***** Calculating the denominator (not the sum) *****  //
       //  *****************************************************  //  
	
	  __syncthreads();              // Wait for all threads to finish the above operation	
	  	
	if (num != Lid){
	
	aux[gid] =   rho[div] * aux[gid];
	//printf("if bid %d, Lid %d,num %d,div %d,tid %d, Did %d,gid %d,z_i[Did] %.4e,z_k[div] %.4e, aux[Did] %.4e\n",bid,Lid,num,div,tid,Did,gid,z_i[Did],z_k[div],aux[gid]);  	
	
	}
	else {
        	if (tid<Rid){   	//  !!!!!!  Warning Warp Divergence !!!!!! 
	 
	aux[gid] =   rho[div] * aux[gid];
	//printf("ifelif bid %d, Lid %d,num %d,div %d,tid %d, Did %d,gid %d,z_i[Did] %.4e,z_k[div] %.4e, aux[Did] %.4e\n",bid,Lid,num,div,tid,Did,gid,z_i[Did],z_k[div],aux[gid]); 	  	
	
	                    }
	                       	    
	}
 
	  __syncthreads();              // Wait for all shuffle reductions per loop	
	  
	  
	// ======>  NEXT Step rearrange to exclude artificial zeros 
	
       // if you want print sum results in order 
 
	
	
	if (gid < N_tracks * N_vertex){
	
	den_h [gid] = aux[Sid] ;
	__syncthreads(); 
	//printf("b %d, L %d,num %d,div %d,t %d, D %d,g %d, Sid %d,div_g %d, rem1 %d, aux[S] %.3e ,den_h [g] %.3e\n",bid,Lid,num,div,tid,Did,gid,Sid,div_g,rem1,aux[Sid],den_h [gid]);  	
	
	}
	
	
	
 }
 
  //*****************************************************************************// 
 //   Kernel parallel reduction for k vertex summation assign data over warps   //
//*****************************************************************************// 
 
 __global__ void par_reduc_for_k_using_warps(float *in, float *aux, int N_warps_per_vertex, int N_tracks, int N_vertex){ 

        // These kernels perform the summation of the same element of i through vertex index k, which means that it is a sum_k for p_ik denominator in DA.
        // This is based on sumBlock_with_shfl_and_par_reduc kernel in shfl_block_sum_v1_git.cu
        // It distributes data over warps (creating sub-blocks) and a 2**n proportional number of big-blocks and performs a parallel reduction
        // 

	int gid = blockIdx.x * blockDim.x + threadIdx.x; 
	int tid = threadIdx.x; 
	int bid = blockIdx.x ;
	int warp =32;
	int num = bid%N_warps_per_vertex;
	int div = bid/N_warps_per_vertex; 
	int Did = num * warp + div * N_tracks + tid ;
	int Lid = (N_warps_per_vertex -1 )% N_warps_per_vertex ;  // Last warp of a group of threads
	int Rid = N_tracks%warp ;  // Last warp of a group of threads
	//int Sid = (tid + (num * warp) + (div * N_warps_per_vertex )) * warp; // 
	//int warp_elem = N_warps_per_vertex * warp * (div+1);      //  variable with maximum element value in each big block
	int N_2n_B = comp_list_2n_blocks(N_vertex);  // Call comp_list_2n_blocks to know the round up to 2**n number of big-blocks for parallel reduction
	//int Pid = num * warp + tid; // variable index to limit the parallel reduction 
	int offset1 =  warp * N_warps_per_vertex * N_2n_B ;
	
	if (tid == 0){
	printf("N_2n_B %d, offset1 %d \n\n",N_2n_B,offset1);
	}
  
  
	  //******************************************************//
         // First set all elements of the auxiliary list to zero //     
	//******************************************************//          
        	
	//out[gid] = 0.0;  
	aux[gid] = 0.0;  
	
	               // Wait for all shuffle reductions per loop
	  
	  	
	__syncthreads();
	  //*******************************************************//
         //             Distributes data over warps               //
	//*******************************************************//	  	
	if (num != Lid){
	
	     //printf("ifb bid %d, Lid %d,num %d,tid %d, Did %d,gid %d,in[Did] %f\n",bid,Lid,num,tid,Did,gid,in[Did]);
	aux[gid] = in[Did];    	
	    // printf("ifa bid %d, Lid %d,num %d,tid %d, Did %d,gid %d,in[Did] %f\n",bid,Lid,num,tid,Did,gid,in[Did]);
	}
	else {
        	if (tid<Rid){   	//  !!!!!!  Warning Warp Divergence !!!!!! 	     
	//printf("elb bid %d, Lid %d,num %d,tid %d, Did %d,gid %d,in[Did] %f\n",bid,Lid,num,tid,Did,gid,in[Did]);		 
	aux[gid] = in[Did];     	
	//printf("ela bid %d, Lid %d,num %d,tid %d, Did %d,gid %d,in[Did] %f\n",bid,Lid,num,tid,Did,gid,in[Did]);	
	                    }
	                       	    
	}
 
//	__syncthreads();              // Wait for all shuffle reductions per loop	
	  
	  
 
		  
	  //*****************************************************//
         //                 Parallel reduction                  //
	//*****************************************************//
	
  __syncthreads();	  
	if (div < N_2n_B/2){   
   __syncthreads();	
  
 int block_offset = N_2n_B/2;
 for (int offset =  warp * N_warps_per_vertex * N_2n_B/2  ; offset >= warp * N_warps_per_vertex; offset /= 2)  {
   __syncthreads();
  aux[gid] += aux[gid+offset];
  
  //__syncthreads();
  
  if (bid < block_offset){
  
  __syncthreads();
 printf("2b %d,t %d, g %d, aux[S] %f, aux[g+o] %f, o %d num %d, N_2n_B/2 %d\n",bid,tid, gid,aux[gid],aux[gid+offset],offset,num, N_2n_B/2);
 }
 block_offset /= 2;
 
 }
      __syncthreads();	
	}	
	 
__syncthreads(); 
	
	
	
	
 }
 
 
 
 
 //************************************************************//
 //**********************                **********************//
 //**********************      MAIN      **********************//
 //**********************                **********************//
 //************************************************************//
 
 int main(void)
{


// host data 
	float *z_i_h, *z_k_h, *sig_h, *p_h, *rho_h, *num_h, *den_h, *aux_h; // host data  
	float beta = 1.0e-3;
	
	//float *a_d, *b_d; // device data
	float *z_i_d, *z_k_d, *sig_d, *p_d, *rho_d, *num_d, *den_d, *aux_d; // device data
	
	int N_tracks = 15, nBytes_data, nBytes_Vertex, i ;
	int MaxNVetex	= 3 , warp = 32;
	int N_vertex = MaxNVetex;

	float Toydata[N_tracks]= {1.0,2.0,3.0,4.0,5.0,15.0,16.0,17.0,18.0,19.0,47.0 , 48.0 , 49.0 , 50.0 , 51.0};

        // Alloc men
	nBytes_data = N_tracks*sizeof(float); 
	int nBytes_par_data = N_tracks*MaxNVetex*sizeof(float); 
	printf("N*MaxNVetex %d \n",N_tracks*MaxNVetex);
	
	// Allocating memory for basics lists
	z_i_h = (float *)malloc(nBytes_data);
	sig_h = (float *)malloc(nBytes_data);
	p_h = (float *)malloc(nBytes_data);
	
	num_h = (float *)malloc(nBytes_par_data);
	den_h = (float *)malloc(nBytes_par_data);	

	nBytes_Vertex = MaxNVetex*sizeof(float); 	
	z_k_h = (float *)malloc(nBytes_Vertex);
	rho_h = (float *)malloc(nBytes_Vertex);	
        z_k_h[0]= 22.1095;  z_k_h[1]= 22.936;  z_k_h[2]= 24.8449;
        
        
        
        // Creating Big list with all warps
        int N_WpBb = N_sub_blocks_warp(N_tracks);// N_WpBb = N. warps per sub block
        int N_Tw = N_blocks_warp(N_tracks,MaxNVetex);// N_Tw = Total n. of warps for all data
        int N_T_threads = N_T_blocks_warp(N_tracks,MaxNVetex); // N_T_threads  = Total n. of elements for all data splited into warp
        int N_Bb_2n = two_to_increase_pow_2(N_vertex) ;// N_Bb_2n = Total n.of big-blocks for parallel reduction
        int N_T_b_2n = N_Bb_2n * N_WpBb; // Total number of blocks for parallel reduction
        int N_T_threads_par_red = two_to_increase_pow_2(N_T_threads);// Total number of elements for parallel reduction (take Tot threads for p_ik and extend it)

        printf("N_WpBb %d, N_Tw %d, N_T_threads %d, N_Bb_2n %d, N_T_Bb_2n %d, N_T_threads_par_red %d\n",N_WpBb,N_Tw,N_T_threads,N_Bb_2n,N_T_b_2n,N_T_threads_par_red);        // Alloc men
        int nBytes_total = N_T_threads*sizeof(float);  // Total amount of memory for the nBytes_par_data for warps size
		
	aux_h = (float *)malloc(nBytes_total);  
        
        

	cudaMalloc((void **) &z_i_d, nBytes_data);	
	cudaMalloc((void **) &sig_d, nBytes_data);	
	cudaMalloc((void **) &p_d, nBytes_data);
	
	cudaMalloc((void **) &num_d, nBytes_par_data); 
	cudaMalloc((void **) &den_d, nBytes_par_data);
	
	cudaMalloc((void **) &rho_d, nBytes_Vertex);
	cudaMalloc((void **) &z_k_d, nBytes_Vertex); 	

	cudaMalloc((void **) &aux_d, nBytes_total); 


	// Host Variables Initialization
	for (i=0; i<N_tracks; i++){
	 
	 z_i_h[i] = Toydata[i];	 
	 sig_h[i] = 1.0;
	 p_h[i] = 1.0;
	 num_h[i] = 0.0; 
	  
	 }
	for (i=0; i<MaxNVetex; i++){	 
	 rho_h[i] = 1.;	  
	 } 
	 

        cudaMemcpy(z_i_d, z_i_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(sig_d, sig_h, nBytes_data, cudaMemcpyHostToDevice);
        cudaMemcpy(p_d, p_h, nBytes_data, cudaMemcpyHostToDevice);
        
        cudaMemcpy(rho_d, rho_h, nBytes_Vertex, cudaMemcpyHostToDevice);  
        
        cudaMemcpy(aux_d, aux_h, nBytes_Vertex, cudaMemcpyHostToDevice);	 

        cudaMemcpy(z_k_d, z_k_h, nBytes_Vertex, cudaMemcpyHostToDevice);	 


//kernel_p_ik_num_Block_warp(float *p_ik, float *z_i, float *z_k, float *sig,float *aux, float beta, int N_warps_per_vertex, int N_tracks, int N_vertex)

                        //  <<<blocks,threads>>> ->   <<<Total N warps,warps>>> 
        kernel_p_ik_pre_Block_warp<<<N_Tw,warp>>>(num_d,den_d,z_i_d,z_k_d,sig_d,rho_d,aux_d,beta,N_WpBb,N_tracks,N_vertex);


	cudaMemcpy(num_h, num_d, nBytes_par_data, cudaMemcpyDeviceToHost); 
	cudaMemcpy(den_h, den_d, nBytes_par_data, cudaMemcpyDeviceToHost);
	
	for (i=0; i< N_tracks; i++)	{ printf("after i  %d  p_ik_h[i] %.10e, %.10e, %.10e \n",i, num_h[i], num_h[i+N_tracks], num_h[i+N_tracks+N_tracks]); }
	printf("\n");
	
	
	//for (i=0; i< N_tracks; i++)	{ printf("after i  %d  den_h[i] %.10e, %.10e, %.10e \n",i, den_h[i], den_h[i+N_tracks], den_h[i+N_tracks+N_tracks]); }
	//printf("\n");
	
	//for (i=0; i< N_tracks*MaxNVetex; i++)	{ printf("%.10e,", num_h[i]); }	
	//printf("\n");
	
	for (i=0; i< N_tracks; i++)	{ printf("soma=> i %d,  p_ik_h[i] %.10e\n",i, num_h[i]+ num_h[i+N_tracks]+ num_h[i+N_tracks+N_tracks]); }
	printf("\n");	
	
	
	  //*****************************************************//
         //                 Parallel reduction                  //
	//*****************************************************//
	   	
	float *aux1_h, *aux1_d;
	int nBytes_par_reduc_3vertex = N_T_threads_par_red*sizeof(float); 
	aux1_h = (float *)malloc(nBytes_par_reduc_3vertex);
	cudaMalloc((void **) &aux1_d, nBytes_par_reduc_3vertex);	
	   
	   
	cudaMemcpy(den_d, den_h, nBytes_par_data, cudaMemcpyHostToDevice);	
	
	//par_reduc_for_k_using_warps(float *in, float *aux,   int N_warps_per_vertex, int N_tracks, int N_vertex)
        
        par_reduc_for_k_using_warps<<<N_T_b_2n,warp>>>(den_d,aux1_d,N_WpBb,N_tracks,N_vertex);	
	
	cudaMemcpy(aux1_h, aux1_d, nBytes_par_reduc_3vertex, cudaMemcpyDeviceToHost); 	
	
	for (i=0; i< N_tracks; i++)	{ printf("after i %d, aux1_h[i] %.10e \n",i, aux1_h[i]); }
	printf("\n");	
	
	

}
 
 
 
 
 
 

#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
#include <iostream>

__global__ void ExampleKernel(float* keys,int N)
{
    // Specialize BlockRadixSort for a 1D block of 128 threads owning 4 integer items each
    typedef cub::BlockRadixSort<float, 4 , 2> BlockRadixSort;
    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    
    float thread_keys[2];
    if(threadIdx.x*2<N) 
    {
   	 thread_keys[0]=keys[threadIdx.x*2];
   	 thread_keys[1]=keys[threadIdx.x*2+1];
 //  	  __syncthreads();
   	 // Collectively sort the keys
   	 //BlockRadixSort(temp_storage).Sort(&thread_keys[threadIdx.x*2]);
     BlockRadixSort(temp_storage).Sort(thread_keys);
     __syncthreads();
   

       keys[threadIdx.x*2]   =thread_keys[0] ;
       keys[threadIdx.x*2+1] =thread_keys[1] ;
       printf(" [%d , %f , %f ] ",threadIdx.x,thread_keys[0],thread_keys[1]);
    }
}

int main()
{
	// Declare, allocate, and initialize device-accessible pointers for sorting data
	const int  num_items =8;           // e.g., 7
	float  h_keys_in[num_items]= {8.0,6.0,7.0,5.0,3.0,0.0,26.26,9.0};         // e.g., [8, 6, 7, 5, 3, 0, 9]
	float  h_keys_out[num_items];        // e.g., [        ...        ]
	
	float  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]

        for(int i=0;i<num_items;i++)
	{
		std::cout<<" i = "<<i<<" : k = "<<h_keys_in[i]<<" \n";
	}
	
	size_t nb=num_items*sizeof(float);
	
	cudaMalloc(&d_keys_in   , nb);

	cudaMemcpy(d_keys_in,h_keys_in, nb , cudaMemcpyHostToDevice);
	
	ExampleKernel<<<1,8>>>(d_keys_in,8);
	cudaMemcpy(h_keys_out,d_keys_in, nb , cudaMemcpyDeviceToHost);
	
	std::cout<<"Sorintg and copy done \n";
	for(int i=0;i<num_items;i++)
	{
		std::cout<<" i = "<<i<<" : k = "<<h_keys_in[i]<<" -> "<<h_keys_out[i]<<" \n";
	}
	

return 0;
}

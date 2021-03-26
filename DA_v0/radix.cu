#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
#include <iostream>

int main()
{
	// Declare, allocate, and initialize device-accessible pointers for sorting data
	int  num_items =7;           // e.g., 7
	float  h_keys_in[7]= {8.0,6.0,7.0,5.0,3.0,0.0,9.0};         // e.g., [8, 6, 7, 5, 3, 0, 9]
	int  h_values_in[7]={1,2,3,4,5,6,7};       // e.g., [0, 1, 2, 3, 4, 5, 6]
	float  h_keys_out[7];        // e.g., [        ...        ]
	int  h_values_out[7];      // e.g., [        ...        ]
	
	int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
	int  *d_keys_out;        // e.g., [        ...        ]
	int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
	int  *d_values_out;      // e.g., [        ...        ]

        for(int i=0;i<7;i++)
	{
		std::cout<<" i = "<<i<<" : k = "<<h_keys_in[i]<<" , v = "<<h_values_in[i]<<" \n";
	}
	
	size_t nb=7*sizeof(float);
	
	cudaMalloc(&d_values_in , nb);
	cudaMalloc(&d_keys_in   , nb);
	cudaMalloc(&d_values_out, nb);
	cudaMalloc(&d_keys_out  , nb);

	cudaMemcpy(d_values_in,h_values_in, nb , cudaMemcpyHostToDevice);
	cudaMemcpy(d_keys_in,h_keys_in, nb , cudaMemcpyHostToDevice);

	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
	    d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
	// Allocate temporary storage
	std::cout<<" tmp storage bytes  = "<<temp_storage_bytes<<"\n";
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
	    d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
	
	cudaMemcpy(h_values_out,d_values_out, nb , cudaMemcpyDeviceToHost);
	cudaMemcpy(h_keys_out,d_keys_out, nb , cudaMemcpyDeviceToHost);
	for(int i=0;i<7;i++)
	{
		std::cout<<" i = "<<i<<" : k = "<<h_keys_out[i]<<" , v = "<<h_values_out[i]<<" \n";
	}
	

return 0;
}

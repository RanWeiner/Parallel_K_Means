#include "K_Means.h"


__device__ double calculate_distance_between_positions_gpu(Position* p1, Position* p2)
{
	double euclid_distance, x, y, z;

	x = p1->x - p2->x;
	y = p1->y - p2->y;
	z = p1->z - p2->z;

	euclid_distance = sqrt(x*x + y*y + z*z);

	return euclid_distance;
}


__global__ void assign_points_to_clusters_gpu(Point* dev_points, int num_of_points, Cluster* dev_clusters, int num_of_clusters, int* dev_point_moved_flag)
{
	double current_distance, min_distance;
	int min_cluster_id, index, i;

	index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < num_of_points)
	{
		min_distance = calculate_distance_between_positions_gpu(&(dev_points[index].position), &(dev_clusters[0].center));
		min_cluster_id = dev_clusters[0].id;

		for (i = 1; i < num_of_clusters; i++)
		{
			current_distance = calculate_distance_between_positions_gpu(&(dev_points[index].position), &(dev_clusters[i].center));

			if (current_distance < min_distance)
			{
				min_distance = current_distance;
				min_cluster_id = dev_clusters[i].id;
			}
		}

		//point moved to another cluster
		if (dev_points[index].cluster_id != min_cluster_id)
		{
			dev_points[index].cluster_id = min_cluster_id;
			*dev_point_moved_flag = 1;
		}
	}
}

__global__ void update_points_position_gpu(Point* dev_points, int num_of_points, double time)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < num_of_points)
	{
		dev_points[index].position.x = dev_points[index].intial_position.x + time*dev_points[index].velocity.vx;
		dev_points[index].position.y = dev_points[index].intial_position.y + time*dev_points[index].velocity.vy;
		dev_points[index].position.z = dev_points[index].intial_position.z + time*dev_points[index].velocity.vz;
	}
}

void init_cuda(Point** dev_points, int num_of_points, Cluster** dev_clusters, int num_of_clusters, int** dev_point_moved_flag)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(DEVICE_ID);
	handle_errors(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n", 0);

	//allocates memory for each process on the GPU
	*dev_points = allocate_points_to_device(num_of_points);
	*dev_clusters = allocate_clusters_to_device(num_of_clusters);
	*dev_point_moved_flag = allocate_point_moved_flag_to_device();
}

int* allocate_point_moved_flag_to_device()
{
	cudaError_t cudaStatus;
	int* dev_point_moved_flag = 0;

	// Allocate GPU buffer for int
	cudaStatus = cudaMalloc((void**)&dev_point_moved_flag, sizeof(int));
	handle_errors(cudaStatus, "cudaMalloc failed!\n", 1, dev_point_moved_flag);

	return dev_point_moved_flag;
}

Cluster* allocate_clusters_to_device(int num_of_clusters)
{
	cudaError_t cudaStatus;
	Cluster* dev_clusters = 0;

	// Allocate GPU buffer for points
	cudaStatus = cudaMalloc((void**)&dev_clusters, num_of_clusters * sizeof(Cluster));
	handle_errors(cudaStatus, "cudaMalloc failed!\n", 1, dev_clusters);

	return dev_clusters;
}

Point* allocate_points_to_device(int num_of_points)
{
	cudaError_t cudaStatus;
	Point* dev_points = NULL;

	// Allocate GPU buffer for points
	cudaStatus = cudaMalloc((void**)&dev_points, num_of_points * sizeof(Point));
	handle_errors(cudaStatus, "cudaMalloc failed!\n", 1, dev_points);

	return dev_points;
}

void update_points_position_by_time(Point* dev_points, Point* points, int num_of_points, double time)
{
	int threads_per_block, num_of_blocks;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	
	cudaGetDeviceProperties(&prop, DEVICE_ID);
	threads_per_block = prop.maxThreadsPerBlock;

	num_of_blocks = (num_of_points + threads_per_block - 1) / threads_per_block;

	// Copy input points array from host memory to GPU buffer
	cudaStatus = cudaMemcpy(dev_points, points, num_of_points * sizeof(Point), cudaMemcpyHostToDevice);
	handle_errors(cudaStatus, "cudaMemcpy host to device failed!\n", 1, dev_points);

	// Launch a kernel on the GPU with blocks & threads
	update_points_position_gpu << <num_of_blocks, threads_per_block >> >(dev_points, num_of_points, time);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	handle_errors(cudaStatus, cudaGetErrorString(cudaStatus), 1, dev_points);

	// cudaDeviceSynchronize waits for the kernel to finish
	cudaStatus = cudaDeviceSynchronize();
	handle_errors(cudaStatus, "cudaDeviceSynchronize returned error code after launching Kernel - update_points_position!\n", 1, dev_points);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, num_of_points * sizeof(Point), cudaMemcpyDeviceToHost);
	handle_errors(cudaStatus, "cudaMemcpy device to host failed!\n", 1, dev_points);

}

void assign_points_to_clusters_gpu(Point* points, Point* dev_points, int num_of_points, Cluster* clusters, Cluster* dev_clusters, int num_of_clusters, int* point_moved_flag, int* dev_point_moved_flag)
{
	int threads_per_block, num_of_blocks;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	
	cudaGetDeviceProperties(&prop, DEVICE_ID);
	threads_per_block = prop.maxThreadsPerBlock;

	num_of_blocks = (num_of_points + threads_per_block - 1) / threads_per_block;

	// Copy input points array from host memory to GPU buffer
	cudaStatus = cudaMemcpy(dev_points, points, num_of_points * sizeof(Point), cudaMemcpyHostToDevice);
	handle_errors(cudaStatus, "cudaMemcpy host to device failed!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);

	// Copy input points array from host memory to GPU buffer
	cudaStatus = cudaMemcpy(dev_clusters, clusters, num_of_clusters * sizeof(Cluster), cudaMemcpyHostToDevice);
	handle_errors(cudaStatus, "cudaMemcpy host to device failed!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);

	// Copy input points array from host memory to GPU buffer
	cudaStatus = cudaMemcpy(dev_point_moved_flag, point_moved_flag, sizeof(int), cudaMemcpyHostToDevice);
	handle_errors(cudaStatus, "cudaMemcpy host to device failed!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);

	// Launch a kernel on the GPU with blocks & threads
	assign_points_to_clusters_gpu << <num_of_blocks, threads_per_block >> >(dev_points, num_of_points, dev_clusters, num_of_clusters, dev_point_moved_flag);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	handle_errors(cudaStatus, cudaGetErrorString(cudaStatus), 3, dev_points, dev_clusters, dev_point_moved_flag);

	// cudaDeviceSynchronize waits for the kernel to finish
	cudaStatus = cudaDeviceSynchronize();
	handle_errors(cudaStatus, "cudaDeviceSynchronize returned error code after launching Kernel - assign_points_to_clusters!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, num_of_points * sizeof(Point), cudaMemcpyDeviceToHost);
	handle_errors(cudaStatus, "cudaMemcpy device to host failed!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(clusters, dev_clusters, num_of_clusters * sizeof(Cluster), cudaMemcpyDeviceToHost);
	handle_errors(cudaStatus, "cudaMemcpy device to host failed!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(point_moved_flag, dev_point_moved_flag, sizeof(int), cudaMemcpyDeviceToHost);
	handle_errors(cudaStatus, "cudaMemcpy device to host failed!\n", 3, dev_points, dev_clusters, dev_point_moved_flag);
}

void end_cuda(Point* dev_my_points, Cluster* dev_clusters, int* dev_point_moved_flag)
{
	cudaError_t cudaStatus;

	//free all cuda allocations
	free_cuda_allocations(3, dev_my_points, dev_clusters, dev_point_moved_flag);

	//reset device
	cudaStatus = cudaDeviceReset();
	handle_errors(cudaStatus, "cudaDeviceReset failed!", 0);
}

void handle_errors(cudaError_t cudaStatus, const char* error, int count, ...)
{
	//function get cuda status , and all the cuda allocations.
	//cuda status will tell us if theres an error, if so we free the allocations

	va_list allocs;
	va_start(allocs, count);
	int i;

	if (cudaStatus != cudaSuccess)
	{
		//free cuda allocations
		for (i = 0; i < count; i++)
			cudaFree(va_arg(allocs, void*));
		va_end(allocs);

		//print the error and abort
		printf("Cuda Error: %s", error);
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
}

void free_cuda_allocations(int count, ...)
{
	//function get the number of allocation and the pointers

	va_list allocs;
	int i;

	va_start(allocs, count);

	//free each allocation
	for (i = 0; i < count; i++)
	{
		cudaFree(va_arg(allocs, void*));
	}

	va_end(allocs);
}









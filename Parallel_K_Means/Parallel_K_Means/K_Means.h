#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASTER 0
#define MIN_NUM_POINTS 10000 
#define MAX_NUM_POINTS 3000000 
#define DEVICE_ID 0


////////////////// [Structs] //////////////////

typedef struct
{
	double x;
	double y;
	double z;
}Position;

typedef struct
{
	double vx;
	double vy;
	double vz;
}Velocity;

typedef struct
{
	int id;
	int num_of_points;
	Position center;
	Position sum_of_points;
	double diameter;
}Cluster;

typedef struct
{
	Position position;
	Position intial_position;
	Velocity velocity;
	int cluster_id;
}Point;

typedef struct
{
	int N, K;
	double T, dT, LIMIT, QM;
}Input;



////////////////// [Prototypes] //////////////////

void parallel_k_means(int rank, int numprocs, const char* input_file_path, const char* output_file_path);
void kmeans(int rank, Cluster* clusters, Input* input, Point* my_points, int num_of_points, Point* dev_points, Cluster* dev_clusters, int* dev_point_moved_flag);
void print_results(double moment, double quality, Cluster* clusters, int num_of_clusters);
void check_allocation(const void *ptr);
void read_data_from_file(const char* file_path, Input* input, Point** points);
void create_MPI_types(MPI_Datatype* MPI_Input_type, MPI_Datatype* MPI_Position_type, MPI_Datatype* MPI_Velocity_type, MPI_Datatype* MPI_Cluster_type, MPI_Datatype* MPI_Point_type);
void create_MPI_Input_type(MPI_Datatype* MPI_Input_type);
void create_MPI_Cluster_type(MPI_Datatype* MPI_Cluster_type, MPI_Datatype* MPI_Position_type);
void create_MPI_Position_type(MPI_Datatype* MPI_Position_type);
void create_MPI_Velocity_type(MPI_Datatype* MPI_Velocity_type);
void create_MPI_Point_type(MPI_Datatype* MPI_Point_type, MPI_Datatype* MPI_Position_type, MPI_Datatype* MPI_Velocity_type);
void free_MPI_types(MPI_Datatype*  MPI_Position_type, MPI_Datatype* MPI_Input_type, MPI_Datatype* MPI_Velocity_type, MPI_Datatype* MPI_Cluster_type, MPI_Datatype* MPI_Point_type);
void write_data_to_file(const char* file_path, double moment, double quality, const Cluster* clusters, int clusters_size);
void free_memory(int rank, Point* all_points, Point* my_points, Cluster* clusters);
void calculate_global_means(Point* my_points, int  num_of_points, Cluster* clusters, int num_of_clusters);
void validate_input(Input* input);
void gather_all_points(Point* points, int points_size, Point* my_points, int my_points_size, int rank, int numprocs, MPI_Datatype* MPI_Point_type);
void set_my_points_portion(int rank, int numprocs, Point* all_points, Point** my_points, int* my_points_size, Input* input, MPI_Datatype* MPI_Point_type);
void clear_local_cluster_data(Cluster* cluster);
double calculate_distance_between_positions(Position* p1, Position* p2);

////////////////// [OpenMP Prototypes] //////////////////

void init_clusters(int rank, Cluster** clusters, Point* points, Input* input);
void update_local_means_data(Point* my_points, int  num_of_points, Cluster* clusters, int num_of_clusters);
void evaluate_clusters_quality(double* quality, Cluster* clusters, Point* points, Input* input);
void calculate_clusters_diameters(Cluster* clusters, Point* points, Input* input);

////////////////// [Cuda Prototypes] //////////////////

void init_cuda(Point** dev_points, int num_of_points, Cluster** dev_clusters, int num_of_clusters, int** dev_point_moved_flag);
Point* allocate_points_to_device(int num_of_points);
Cluster* allocate_clusters_to_device(int num_of_clusters);
int* allocate_point_moved_flag_to_device();
void update_points_position_by_time(Point* dev_points, Point* points, int num_of_points, double time);
void assign_points_to_clusters_gpu(Point* points, Point* dev_points, int num_of_points, Cluster* clusters, Cluster* dev_clusters, int num_of_clusters, int* point_moved_flag, int* dev_point_moved_flag);
void handle_errors(cudaError_t cudaStatus, const char* error, int count, ...);
void end_cuda(Point* dev_my_points, Cluster* dev_clusters, int* dev_point_moved_flag);
void free_cuda_allocations(int count, ...);

__device__ double calculate_distance_between_positions_gpu(Position* p1, Position* p2);
__global__ void assign_points_to_clusters_gpu(Point* dev_points, int num_of_points, Cluster* dev_clusters, int num_of_clusters, int* dev_point_moved_flag);
__global__ void update_points_position_gpu(Point* dev_points, int num_of_points, double time);




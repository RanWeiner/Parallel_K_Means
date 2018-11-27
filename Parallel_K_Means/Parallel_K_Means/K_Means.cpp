#pragma warning(disable: 4996)
#include "K_Means.h"


int main(int argc, char *argv[])
{
	//change the INPUT & OUTPUT files path 
	const char* input_file_path = "D:\\INPUT_FILE.txt";
	const char* output_file_path = "D:\\OUTPUT_FILE.txt";

	int rank, numprocs;
	double start_time, end_time;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (numprocs < 3)
	{
		printf("num of processes must be at least three.\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	if (rank == MASTER)
	{
		start_time = MPI_Wtime();
	}

	parallel_k_means(rank, numprocs, input_file_path, output_file_path);


	if (rank == MASTER)
	{
		end_time = MPI_Wtime();
		printf("//////[k means finished in %lf]//////\n\n", end_time - start_time);
		fflush(stdout);
	}

	MPI_Finalize();
	return 0;
}

void parallel_k_means(int rank, int numprocs, const char* input_file_path, const char* output_file_path)
{
	Point* all_points = NULL, *my_points = NULL, *dev_my_points = NULL;
	Cluster* clusters = NULL, *dev_clusters = NULL;
	Input input;
	int my_num_of_points, quality_achived = 0, n;
	double quality = 0, time = 0, iterations;
	MPI_Datatype MPI_Point_type, MPI_Input_type, MPI_Cluster_type, MPI_Position_type, MPI_Velocity_type;
	int* dev_point_moved_flag = NULL;

	//create MPI types
	create_MPI_types(&MPI_Input_type, &MPI_Position_type, &MPI_Velocity_type, &MPI_Cluster_type, &MPI_Point_type);

	if (rank == MASTER)
	{
		//master read & initialize all the data from the file 
		read_data_from_file(input_file_path, &input, &all_points);
		init_clusters(rank, &clusters, all_points, &input);

		//master send slaves the input parameters given and the clusters
		MPI_Bcast(&input, 1, MPI_Input_type, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(clusters, input.K, MPI_Cluster_type, MASTER, MPI_COMM_WORLD);
	}
	else
	{
		//slaves recieve the parameters
		MPI_Bcast(&input, 1, MPI_Input_type, MASTER, MPI_COMM_WORLD);

		//slaves initialize cluster array & recieve clusters from master
		init_clusters(rank, &clusters, all_points, &input);
		MPI_Bcast(clusters, input.K, MPI_Cluster_type, MASTER, MPI_COMM_WORLD);
	}

	//each process gets point portion from master
	set_my_points_portion(rank, numprocs, all_points, &my_points, &my_num_of_points, &input, &MPI_Point_type);

	//set cuda device & allocate memory for each proccess on the gpu
	init_cuda(&dev_my_points, my_num_of_points, &dev_clusters, input.K, &dev_point_moved_flag);

	//set number of iterations for the k means algorithm
	iterations = (input.T * 1.0) / input.dT;

	for (n = 0; n <= iterations && !quality_achived; n++)
	{
		//update time
		time = n*input.dT;

		//update points position by the time - using Cuda
		update_points_position_by_time(dev_my_points, my_points, my_num_of_points, time);

		//perform kmeans until no point changed cluster in the world or max number of iterations exceeded
		kmeans(rank, clusters, &input, my_points, my_num_of_points, dev_my_points, dev_clusters, dev_point_moved_flag);

		//gather all points from each process
		gather_all_points(all_points, input.N, my_points, my_num_of_points, rank, numprocs, &MPI_Point_type);

		if (rank == MASTER)
		{
			//master evaluate the quality of the clusters, if quality achived we are done
			evaluate_clusters_quality(&quality, clusters, all_points, &input);
			quality_achived = (quality < input.QM) ? 1 : 0;
		}

		//master send all processes if quality achived in order to finish or continue loop
		MPI_Bcast(&quality_achived, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	//master write results to output file
	if (rank == MASTER)
	{
		write_data_to_file(output_file_path, time, quality, clusters, input.K);
		//print_results(time, quality, clusters, input.K);
	}

	//free all cuda memory allocated & reset device
	end_cuda(dev_my_points, dev_clusters, dev_point_moved_flag);

	//free all cpu alocated memory
	free_memory(rank, all_points, my_points, clusters);
	free_MPI_types(&MPI_Position_type, &MPI_Input_type, &MPI_Velocity_type, &MPI_Cluster_type, &MPI_Point_type);
}

void validate_input(Input* input)
{
	if (input->N < MIN_NUM_POINTS || input->N > MAX_NUM_POINTS)
	{
		printf("num of points is incorrect.\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	if (input->K > input->N)
	{
		printf("number of clusters is bigger then the number of points.\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
}


void calculate_clusters_diameters(Cluster* clusters, Point* points, Input* input)
{
	int max_threads, shared_size, shared_index = 0, i,j=0;
	double* shared_clusters_max_points_distance = NULL;
	double distance = 0;

	//get the max num of cores
	max_threads = omp_get_max_threads();

	//set the omp clusters max distances array - the array contains all the max point distances of each cluster
	shared_size = input->K * max_threads;
	shared_clusters_max_points_distance = (double*)calloc(shared_size, sizeof(double));
	check_allocation(shared_clusters_max_points_distance);
	
	//set the num of threads for the parallel region
	omp_set_num_threads(max_threads);

#pragma omp parallel for private(shared_index,j,distance)
	for (i = 0; i < input->N-1 ; i++)
	{
		//the index in the shared threads array
		shared_index = omp_get_thread_num() * input->K + points[i].cluster_id;

		for (j = i + 1; j < input->N; j++)
		{
			//check distance with points in the same cluster
			if (points[i].cluster_id == points[j].cluster_id)
			{
				distance = calculate_distance_between_positions(&(points[i].position), &(points[j].position));

				if (distance > shared_clusters_max_points_distance[shared_index])
				{
					shared_clusters_max_points_distance[shared_index] = distance;
				}
			}
		}
	}

	//merge shared clusters array into the original clusters array 
	for (int i = 0; i < shared_size; i++)
	{
		if (shared_clusters_max_points_distance[i] > clusters[i%input->K].diameter)
		{
			clusters[i%input->K].diameter = shared_clusters_max_points_distance[i];
		}
	}

	//free allocation
	free(shared_clusters_max_points_distance);
}


void evaluate_clusters_quality(double* quality, Cluster* clusters, Point* points, Input* input)
{
	int i, j = 0;
	double sum_quality = 0;

	calculate_clusters_diameters(clusters, points, input);

#pragma omp parallel for private(j) reduction(+:sum_quality)
	for (i = 0; i < input->K; i++)
	{
		for (j = 0; j < input->K; j++)
		{
			if (i != j)
			{
				sum_quality += clusters[i].diameter / calculate_distance_between_positions(&(clusters[i].center), &(clusters[j].center));
			}
		}
		//reseting diameter in case quality is not reached
		clusters[i].diameter = 0; 
	}

	*quality = sum_quality / (input->K * (input->K - 1));
}


void init_clusters(int rank, Cluster** clusters, Point* points, Input* input)
{
	int i;
	*clusters = (Cluster*)malloc(sizeof(Cluster)*(input->K));
	check_allocation(*clusters);

	if (rank == MASTER)
	{
		//set initial K points as clusters centers
#pragma omp parallel for
		for (i = 0; i < input->K; i++)
		{
			(*clusters)[i].id = i;
			(*clusters)[i].center.x = points[i].position.x;
			(*clusters)[i].center.y = points[i].position.y;
			(*clusters)[i].center.z = points[i].position.z;
			(*clusters)[i].num_of_points = 0;
			(*clusters)[i].diameter = 0;
			(*clusters)[i].sum_of_points.x = 0;
			(*clusters)[i].sum_of_points.y = 0;
			(*clusters)[i].sum_of_points.z = 0;
		}
	}
}

void set_my_points_portion(int rank, int numprocs, Point* all_points, Point** my_points, int* my_points_size, Input* input, MPI_Datatype* MPI_Point_type)
{
	int remainder, i, slave_portion;
	MPI_Status status;

	//divide the data equally between processes
	slave_portion = input->N / numprocs;
	remainder = input->N % numprocs;

	//all processes working on the same amount of data, the master do the extra work if needed
	*my_points_size = (rank == MASTER) ? (slave_portion + remainder) : slave_portion;

	//each process allocate memory
	*my_points = (Point*)malloc(sizeof(Point)*(*my_points_size));
	check_allocation(*my_points);

	//master set his points and then send each process his part of data - using OMP
	if (rank == MASTER)
	{
		memcpy(*my_points, all_points, (*my_points_size) * sizeof(Point));

		for (i = 0; i < numprocs - 1; i++)
		{
			MPI_Send(all_points + (*my_points_size) + i*slave_portion, slave_portion, *MPI_Point_type, i + 1, 0, MPI_COMM_WORLD);
		}
	}
	//each process recieve part of data from master process
	else
	{
		MPI_Recv(*my_points, *my_points_size, *MPI_Point_type, 0, 0, MPI_COMM_WORLD, &status);
	}
}

void kmeans(int rank, Cluster* clusters, Input* input, Point* my_points, int num_of_points, Point* dev_points, Cluster* dev_clusters, int* dev_point_moved_flag)
{
	int local_point_moved_flag, iter = 0, global_point_moved_flag = 0, global_num_of_points = 0;

	do
	{
		local_point_moved_flag = 0;

		//assign each point to a cluster - using Cuda
		assign_points_to_clusters_gpu(my_points, dev_points, num_of_points, clusters, dev_clusters, input->K, &local_point_moved_flag, dev_point_moved_flag);

		//sum all the point moved flag from each process in order to know whether point changed cluster in the world
		MPI_Allreduce(&local_point_moved_flag, &global_point_moved_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//update each process clusters data
		update_local_means_data(my_points, num_of_points, clusters, input->K);

		//calculate the new cluster centers with the local clusters data
		calculate_global_means(my_points, num_of_points, clusters, input->K);

		iter++;

	} while (global_point_moved_flag> 0 && iter < input->LIMIT);
}

void gather_all_points(Point* points, int points_size, Point* my_points, int my_points_size, int rank, int numprocs, MPI_Datatype* MPI_Point_type)
{
	int i, slave_portion;
	MPI_Status status;

	if (rank == MASTER)
	{
		//master copy his points
		memcpy(points, my_points, sizeof(Point)*my_points_size);

		//master recieve points from slaves
		slave_portion = points_size / numprocs;
		for (i = my_points_size; i < points_size; i += slave_portion)
		{
			MPI_Recv(&points[i], slave_portion, *MPI_Point_type, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}
	}

	else
	{
		//slave send his points portion to master
		MPI_Send(my_points, my_points_size, *MPI_Point_type, MASTER, 0, MPI_COMM_WORLD);
	}
}

void free_MPI_types(MPI_Datatype*  MPI_Position_type, MPI_Datatype* MPI_Input_type, MPI_Datatype* MPI_Velocity_type, MPI_Datatype* MPI_Cluster_type, MPI_Datatype* MPI_Point_type)
{
	MPI_Type_free(MPI_Position_type);
	MPI_Type_free(MPI_Input_type);
	MPI_Type_free(MPI_Velocity_type);
	MPI_Type_free(MPI_Cluster_type);
	MPI_Type_free(MPI_Point_type);
}

void print_results(double moment, double quality, Cluster* clusters, int num_of_clusters)
{
	int i;

	printf("First occurrence t = %lf with q = %lf\n\n", moment, quality);

	printf("Centers of the clusters :\n\n");

	for (i = 0; i < num_of_clusters; i++)
	{
		printf("%lf\t%lf\t%lf\n", clusters[i].center.x, clusters[i].center.y, clusters[i].center.z);
	}
}

void calculate_global_means(Point* my_points, int  num_of_points, Cluster* clusters, int num_of_clusters)
{
	int i, global_num_of_points;
	double global_sum_x, global_sum_y, global_sum_z;

	for (i = 0; i < num_of_clusters; i++)
	{
		//each process gets the global num of points of the cluster
		MPI_Allreduce(&clusters[i].num_of_points, &global_num_of_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//if no points assign to a cluster no need calculate
		if (global_num_of_points != 0)
		{
			//all process recieve the clusters sum of points in all dimenstions
			MPI_Allreduce(&clusters[i].sum_of_points.x, &global_sum_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&clusters[i].sum_of_points.y, &global_sum_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&clusters[i].sum_of_points.z, &global_sum_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			//set the cluster new center
			clusters[i].center.x = global_sum_x / (global_num_of_points*1.0);
			clusters[i].center.y = global_sum_y / (global_num_of_points*1.0);
			clusters[i].center.z = global_sum_z / (global_num_of_points*1.0);

			//clear cluster data for the next iterations
			clear_local_cluster_data(&clusters[i]);
		}
	}
}

void clear_local_cluster_data(Cluster* cluster)
{
	cluster->num_of_points = 0;
	cluster->sum_of_points.x = 0;
	cluster->sum_of_points.y = 0;
	cluster->sum_of_points.z = 0;
}

void update_local_means_data(Point* my_points, int num_of_points, Cluster* clusters, int num_of_clusters)
{
	int max_threads, shared_size, shared_index = 0, i;
	Cluster* shared_clusters = NULL;

	//get the max num of cores
	max_threads = omp_get_max_threads();

	//set the omp clusters array
	shared_size = num_of_clusters * max_threads;
	shared_clusters = (Cluster*)calloc(shared_size, sizeof(Cluster));
	check_allocation(shared_clusters);

	//set the num of threads for the parallel region
	omp_set_num_threads(max_threads);

	//update clusters num of points & the sum of points position
#pragma omp parallel for private(shared_index)
	for (i = 0; i < num_of_points; i++)
	{
		//set the omp array index
		shared_index = omp_get_thread_num() * num_of_clusters + my_points[i].cluster_id;

		//update shared clusters array
		shared_clusters[shared_index].num_of_points++;
		shared_clusters[shared_index].sum_of_points.x += my_points[i].position.x;
		shared_clusters[shared_index].sum_of_points.y += my_points[i].position.y;
		shared_clusters[shared_index].sum_of_points.z += my_points[i].position.z;
	}


	//merge shared clusters array into the original clusters array 
	for (int i = 0; i < shared_size; i++)
	{
		clusters[i%num_of_clusters].num_of_points += shared_clusters[i].num_of_points;
		clusters[i%num_of_clusters].sum_of_points.x += shared_clusters[i].sum_of_points.x;
		clusters[i%num_of_clusters].sum_of_points.y += shared_clusters[i].sum_of_points.y;
		clusters[i%num_of_clusters].sum_of_points.z += shared_clusters[i].sum_of_points.z;
	}

	free(shared_clusters);
}



double calculate_distance_between_positions(Position* p1, Position* p2)
{
	double euclid_distance, x, y, z;

	x = p1->x - p2->x;
	y = p1->y - p2->y;
	z = p1->z - p2->z;

	euclid_distance = sqrt(x*x + y*y + z*z);

	return euclid_distance;
}

void read_data_from_file(const char* file_path, Input* input, Point** points)
{
	FILE* file;
	int i;
	double x, y, z, vx, vy, vz;

	//open file to read
	file = fopen(file_path, "r");

	//file not opened
	if (file == NULL)
	{
		printf("file not opened - make sure the .txt file is in the correct path");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	//read first line and set parameters
	fscanf(file, "%d%d%lf%lf%lf%lf", &(input->N), &(input->K), &(input->T), &(input->dT), &(input->LIMIT), &(input->QM));

	//check if the input from the text file is valid
	validate_input(input);

	//initialize points array by the given number of points
	*points = (Point*)malloc(sizeof(Point)*(input->N));
	check_allocation(*points);

	//read input parameters and initialize all the points 
	for (i = 0; i < (input->N); i++)
	{
		//read one point data
		fscanf(file, "%lf%lf%lf%lf%lf%lf", &x, &y, &z, &vx, &vy, &vz);

		//set point initial position & position
		(*points)[i].position.x = (*points)[i].intial_position.x = x;
		(*points)[i].position.y = (*points)[i].intial_position.y = y;
		(*points)[i].position.z = (*points)[i].intial_position.z = z;

		//set point velocity
		(*points)[i].velocity.vx = vx;
		(*points)[i].velocity.vy = vy;
		(*points)[i].velocity.vz = vz;

		//set cluster id to -1 , unassigned yet
		(*points)[i].cluster_id = -1;
	}

	//close the file
	fclose(file);
}

void write_data_to_file(const char* file_path, double moment, double quality, const Cluster* clusters, int clusters_size)
{
	int i;
	FILE* file;

	//open file to write
	file = fopen(file_path, "w");

	//file not opened
	if (file == NULL)
	{
		printf("file failed to open.");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	//write to the file the result
	fprintf(file, "First occurrence t = %lf with q = %lf\n\n", moment, quality);
	fprintf(file, "Centers of the clusters :\n\n");

	//write to the file all the clusters centers
	for (i = 0; i < clusters_size; i++)
	{
		fprintf(file, "%lf\t%lf\t%lf\n", clusters[i].center.x, clusters[i].center.y, clusters[i].center.z);
	}

	//close the file
	fclose(file);
}

void check_allocation(const void *ptr)
{
	if (!ptr)
	{
		puts("\nAllocation failed!\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
}

void create_MPI_types(MPI_Datatype* MPI_Input_type, MPI_Datatype* MPI_Position_type, MPI_Datatype* MPI_Velocity_type, MPI_Datatype* MPI_Cluster_type, MPI_Datatype* MPI_Point_type)
{
	//create all the MPI types
	create_MPI_Input_type(MPI_Input_type);
	create_MPI_Position_type(MPI_Position_type);
	create_MPI_Velocity_type(MPI_Velocity_type);
	create_MPI_Cluster_type(MPI_Cluster_type, MPI_Position_type);
	create_MPI_Point_type(MPI_Point_type, MPI_Position_type, MPI_Velocity_type);
}

void create_MPI_Point_type(MPI_Datatype* MPI_Point_type, MPI_Datatype* MPI_Position_type, MPI_Datatype* MPI_Velocity_type)
{
	Point point;
	const int nitems = 4;
	MPI_Datatype type[nitems] = { *MPI_Position_type, *MPI_Position_type, *MPI_Velocity_type, MPI_INT };
	int blocklen[nitems] = { 1,1,1,1 };
	MPI_Aint disp[nitems];

	disp[0] = (char*)&point.position - (char*)&point;
	disp[1] = (char*)&point.intial_position - (char*)&point;
	disp[2] = (char*)&point.velocity - (char*)&point;
	disp[3] = (char*)&point.cluster_id - (char*)&point;

	MPI_Type_create_struct(nitems, blocklen, disp, type, MPI_Point_type);
	MPI_Type_commit(MPI_Point_type);
}

void create_MPI_Velocity_type(MPI_Datatype* MPI_Velocity_type)
{
	Velocity velocity;
	const int nitems = 3;
	MPI_Datatype type[nitems] = { MPI_DOUBLE , MPI_DOUBLE , MPI_DOUBLE };
	int blocklen[nitems] = { 1,1,1 };
	MPI_Aint disp[nitems];

	disp[0] = (char*)&velocity.vx - (char*)&velocity;
	disp[1] = (char*)&velocity.vy - (char*)&velocity;
	disp[2] = (char*)&velocity.vz - (char*)&velocity;

	MPI_Type_create_struct(nitems, blocklen, disp, type, MPI_Velocity_type);
	MPI_Type_commit(MPI_Velocity_type);
}

void create_MPI_Position_type(MPI_Datatype* MPI_Position_type)
{
	Position position;
	const int nitems = 3;
	MPI_Datatype type[nitems] = { MPI_DOUBLE , MPI_DOUBLE , MPI_DOUBLE };
	int blocklen[nitems] = { 1,1,1 };
	MPI_Aint disp[nitems];

	disp[0] = (char*)&position.x - (char*)&position;
	disp[1] = (char*)&position.y - (char*)&position;
	disp[2] = (char*)&position.z - (char*)&position;

	MPI_Type_create_struct(nitems, blocklen, disp, type, MPI_Position_type);
	MPI_Type_commit(MPI_Position_type);
}

void create_MPI_Cluster_type(MPI_Datatype* MPI_Cluster_type, MPI_Datatype* MPI_Position_type)
{
	Cluster cluster;
	const int nitems = 5;
	MPI_Datatype type[nitems] = { MPI_INT,MPI_INT, *MPI_Position_type, *MPI_Position_type, MPI_DOUBLE };
	int blocklen[nitems] = { 1,1,1,1,1 };
	MPI_Aint disp[nitems];

	disp[0] = (char*)&cluster.id - (char*)&cluster;
	disp[1] = (char*)&cluster.num_of_points - (char*)&cluster;
	disp[2] = (char*)&cluster.center - (char*)&cluster;
	disp[3] = (char*)&cluster.sum_of_points - (char*)&cluster;
	disp[4] = (char*)&cluster.diameter - (char*)&cluster;

	MPI_Type_create_struct(nitems, blocklen, disp, type, MPI_Cluster_type);
	MPI_Type_commit(MPI_Cluster_type);
}

void create_MPI_Input_type(MPI_Datatype* MPI_Input_type)
{
	Input input;
	const int nitems = 6;
	MPI_Datatype type[nitems] = { MPI_INT,MPI_INT, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[nitems] = { 1,1,1,1,1,1 };
	MPI_Aint disp[nitems];

	disp[0] = (char*)&input.N - (char*)&input;
	disp[1] = (char*)&input.K - (char*)&input;
	disp[2] = (char*)&input.T - (char*)&input;
	disp[3] = (char*)&input.dT - (char*)&input;
	disp[4] = (char*)&input.LIMIT - (char*)&input;
	disp[5] = (char*)&input.QM - (char*)&input;

	MPI_Type_create_struct(nitems, blocklen, disp, type, MPI_Input_type);
	MPI_Type_commit(MPI_Input_type);
}

void free_memory(int rank, Point* all_points, Point* my_points, Cluster* clusters)
{
	if (rank == MASTER)
	{
		free(all_points);
	}

	free(my_points);
	free(clusters);
}


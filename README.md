# Parallel-K_MEANS
Parallel implementation of the K-Means algorithm using OpenMP, MPI and CUDA.

The project was developed as a part of the Parallel & Distributed Computation course.

to run this project you will need Nvidia's GPU and mpich http://www.mpich.org/static/downloads/1.4.1p1/.
 In the main function located in "K_Means.cpp" change the input path to the input file (sample for input file included "input.txt").
 Also change the output path to your desired directory (the results will be stored in a text file).
Build solution and then open wmpichexec, browse to your solution x64/debug/.exe file and set number of processes to 3 or above. 


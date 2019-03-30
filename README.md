# Parallel Programming 

This project is a set of three parallel computing problems implemented in C/C++. 

* The 1st goal  is parallelizing a block of code using **MPI** respecting any constrains inserted by the code. GaussSeidel ALgorithm was implemented.  [Source Code](https://gitlab.com/timos/parallel-programming-/tree/master/Mpi%20Framework/Codes)  
* The 2nd goal  is accelerating  matrix multiplication operations using **GPU**. For this part **CUDA** was used. [Source Code](https://github.com/Stamatios-Korres/Parallel-Programming-Cpp/tree/master/GPU%20(Cuda)%20Programming) 
* The 3rd objective is to **synchronize**  processes and threads in a multicore environment using locks,mutexes,  and other optimization and synchronization tools. [Source Code](https://gitlab.com/timos/parallel-programming-/tree/master/Synchronization%20in%20Multicore%20Systems/Source%20Code)

### Technologies Used

* C/C++ 
* [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [MPI](https://www.open-mpi.org/)
* [Locks](http://www.cplusplus.com/reference/mutex/mutex/lock/)
* [TBBs](https://www.threadingbuildingblocks.org/)

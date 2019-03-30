#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include "utils.h"

int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2];	 //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];  	 //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];           	 //processor grid dimensions
    int i,j,t;
    int global_converged=0,converged=0; 	//flags for convergence, global and per process
    MPI_Datatype dummy;     			//dummy datatype used to align user-defined datatypes in memory
    // double omega; 				//relaxation factor - useless for Jacobi

    struct timeval tts,ttf,tcs,tcf;  		//Timers: total-tts,ttf, computation-tcs,tcf
    double ttotal=0,tcomp=0,total_time,comp_time, tconv=0,conv_time;
    double ** U, ** u_current, ** u_previous, ** swap;
 
	//Global matrix, local current and previous matrices, pointer to swap between current and previous
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	//-----------πλέον το size έχει το συνολικό πλήθος των διεργασιών , ενώ το rank είναι το pid()--------//
	//----Read 2D-domain dimensions and process grid dimensions from stdin----//
	if (argc!=5) {
        	fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        	exit(-1);
	}
        else{
	        global[0]=atoi(argv[1]);
        	global[1]=atoi(argv[2]);
        	grid[0]=atoi(argv[3]);
        	grid[1]=atoi(argv[4]);
	}

    //----Create 2D-cartesian communicator----//
	//----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2]={0,0};       //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator
		
    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    //communicator creation
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);	      //rank mapping on the new communicator
    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//
	
    
    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
        }
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
        }
    }

	//Initialization of omega
   // omega=2.0/(1+sin(3.14/global[0]));

    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
	
	if (rank==0) {
		U=allocate2d(global_padded[0],global_padded[1]);  
        	init2d(U,global[0],global[1]);
	}

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

	u_previous=allocate2d(local[0]+2,local[1]+2);
	u_current=allocate2d(local[0]+2,local[1]+2);  
 
	MPI_Datatype global_block;	
  	MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
  	MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
  	MPI_Type_commit(&global_block);
	//----Datatype definition for the 2D-subdomain on the local matrix----//
	MPI_Datatype local_block;
	MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);    
	MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
	MPI_Type_commit(&local_block);
	//----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//

	int * scatteroffset, * scattercounts;
 	if (rank==0) {
        	scatteroffset=(int*)malloc(size*sizeof(int));
        	scattercounts=(int*)malloc(size*sizeof(int));
        	for (i=0;i<grid[0];i++){
        	    for (j=0;j<grid[1];j++) {
        	        scattercounts[i*grid[1]+j]= 1;  
        	        scatteroffset[i*grid[1]+j]= (local[0]*local[1]*grid[1]*i+local[1]*j);
        	    }		
    		}
	}
	double *ptr;
	if(rank!=0)
		ptr = malloc(1);
	else if(rank==0)
		ptr = &U[0][0];		
	MPI_Scatterv(ptr,scattercounts,scatteroffset,global_block,&(u_previous[1][1]),1,local_block,0,MPI_COMM_WORLD);
	MPI_Scatterv(ptr,scattercounts,scatteroffset,global_block,&(u_current[1][1]),1,local_block,0,MPI_COMM_WORLD);	
		if(rank==0)
			free2d(U,global_padded[0],global_padded[1]);
		else
			free(ptr);
	
	/*---------------------------------------------------------------------------------------------------------*/
	//columns
	MPI_Datatype col_block;
	MPI_Type_vector(local[0]+2,1,local[1]+2,MPI_DOUBLE,&dummy);
	MPI_Type_create_resized(dummy,0,sizeof(double),&col_block);
	MPI_Type_commit(&col_block);
	//rows
	MPI_Datatype row_block;
	MPI_Type_contiguous(local[1]+2,MPI_DOUBLE,&row_block);
	MPI_Type_commit(&row_block);
	
	int up,down,left,right;
	MPI_Cart_shift(CART_COMM,1,1,&left,&right);
	MPI_Cart_shift(CART_COMM,0,1,&up,&down);
	if(up>=rank || up <0)
		up=(-1);
	if(down<=rank|| down <0)
		down=(-1);
	if(left>=rank|| left <0)
		left=(-1);
	if(right<=rank|| right <0)
		right=(-1);
	int i_min,i_max,j_min,j_max;
	int check_pad[2] = {0,0};
	check_pad[0] = global_padded[0]-global[0];
	check_pad[1] = global_padded[1]-global[1];
	check_iterations(&i_min,&i_max,&j_max,&j_min,up,down,left,right,check_pad,local);
	//Corner Case //
	if(check_pad[0] > local[0]){
		int up1,down1;
		MPI_Cart_shift(CART_COMM,0,2,&up1,&down1);	
		if(down1<0 && down>=0)
			i_max = check_pad[0]-local[0]+1;
	}
	if(check_pad[1] > local[1]){
		int left1,right1;
		MPI_Cart_shift(CART_COMM,1,2,&left1,&right1);	
		if(right1 <0 && right>=0)
			j_max = check_pad[1]-local[1]+1;
	}	
	int counter=0;
	MPI_Status array_of_statuses[8];
	gettimeofday(&tts,NULL);
	#define TEST_CONV
	#ifdef TEST_CONV
	struct timeval tcns,tcnf;  		//Timers: converge-tcns,tcnf
	for (t=0;t<T && !global_converged;t++) {
	#endif
	#ifndef TEST_CONV
	#undef T
	#define T 256
	for (t=0;t<T;t++) {
	#endif
		counter=0;
		MPI_Request requests[8]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};	
	
		if(right >= 0){
			MPI_Isend(&(u_previous[0][local[1]]),1,col_block,right,5,MPI_COMM_WORLD,&requests[counter]);
			counter++;	
			MPI_Irecv(&(u_previous[0][local[1]+1]),1,col_block,right,4,MPI_COMM_WORLD,&requests[counter]);
			counter++;
		}
		if(left >= 0){	
			MPI_Irecv(&(u_previous[0][0]),1,col_block,left,5,MPI_COMM_WORLD,&requests[counter]);
			counter++;		 				
			MPI_Isend(&(u_previous[0][1]),1,col_block,left,4,MPI_COMM_WORLD,&requests[counter]);	
			counter++;
		}
		if(up>=0){
			MPI_Irecv(&(u_previous[0][0]),1,row_block,up,3,MPI_COMM_WORLD,&requests[counter]);
			counter++;
			MPI_Isend(&(u_previous[1][0]),1,row_block,up,2,MPI_COMM_WORLD,&requests[counter]);
			counter++;
		}
		if(down>=0){
			MPI_Isend(u_previous[local[0]],1,row_block,down,3,MPI_COMM_WORLD,&requests[counter]);
			counter++;
			MPI_Irecv(u_previous[local[0]+1],1,row_block,down,2,MPI_COMM_WORLD,&requests[counter]);
			counter++;
		}
		MPI_Waitall(counter,requests,array_of_statuses);
		gettimeofday(&tcs,NULL);
		for(i=i_min;i<i_max;i++){
			for(j=j_min;j<j_max;j++){
				u_current[i][j] = (u_previous[i-1][j]+u_previous[i+1][j]+u_previous[i][j-1]+u_previous[i][j+1])/4;				
			}
		}
		#ifdef TEST_CONV
		gettimeofday(&tcns,NULL);
		if (t%C==0) {
			converged=converge(u_previous,u_current,local[0]+1,local[1]+1);	
			//gettimeofday(&tcs,NULL);
			MPI_Allreduce(&converged,&global_converged,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
			//gettimeofday(&tcf,NULL);
			//tcomp+=(tcf.tv_sec-tcs.tv_sec)+(tcf.tv_usec-tcs.tv_usec)*0.000001;
			if(global_converged == size)
				global_converged=1;
			else
				global_converged=0;
		}
		gettimeofday(&tcnf,NULL);
		tconv+=(tcnf.tv_sec-tcns.tv_sec)+(tcnf.tv_usec-tcns.tv_usec)*0.000001; 
		#endif
		swap=u_previous;
		u_previous=u_current;
		u_current=swap;
		gettimeofday(&tcf,NULL);
		tcomp+=(tcf.tv_sec-tcs.tv_sec)+(tcf.tv_usec-tcs.tv_usec)*0.000001; 
	}
	gettimeofday(&ttf,NULL);
	ttotal=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;
	MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	#ifdef TEST_CONV
	MPI_Reduce(&tconv,&conv_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	#endif
	#ifndef TEST_CONV
	conv_time=0;
	#endif

	//----Rank 0 gathers local matrices back to the global matrix----//
	if (rank==0) {
        	U=allocate2d(global_padded[0],global_padded[1]);
	}
	if(rank!=0)
		ptr = malloc(1);
	else if(rank==0)
		ptr = &U[0][0];	
	MPI_Gatherv(&(u_previous[1][1]),1,local_block,ptr,scattercounts,scatteroffset,global_block,0,MPI_COMM_WORLD);

	//*************TODO*******************//
	/*Fill your code here*/
	free(u_current);
	free(u_previous);
	// #define PRINT_RESULTS
     	//----Printing results----//

	//**************TODO: Change "Jacobi" to "GaussSeidelSOR" or "RedBlackSOR" for appropriate printing****************//
    if (rank==0) {
       printf("\nJacobi X %d Y %d Px %d Py %d Iter %d ConvergeTime %lf ComputationTime %lf TotalTime %lf midpoint %lf\n",global[0],global[1],grid[0],grid[1],t-1,conv_time,comp_time,total_time,U[global[0]/2][global[1]/2]);
	
        #ifdef PRINT_RESULTS
        char * s=malloc(50*sizeof(char));
        sprintf(s,"resJacobiMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
        fprint2d(s,U,global[0],global[1]);
     	free(s);
        #endif
    }
	MPI_Finalize();
	return 0;
}

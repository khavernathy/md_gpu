/* ==================================================================
	Programmers: Alfredo Peguero Tejada & Douglas Franz
	A molecular dynamics NVE code for GPU.
	To compile: nvcc my_file.cu -o my_exe in the rc machines
	run with, e.g. ./my_exe 
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <map>
#include "constants.cpp"
#include "system.cpp"
#include <vector>


/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double px, py, pz, vx, vy, vz, ax, ay, az, fx, fy, fz, charge, mass, LJsig, LJeps;
	char name[2];
} atom;

atom * atom_list;		/* list of all data points  for GPU             */
atom * h_atom_list;		/* for host */

// These are for an old way of tracking time 
struct timezone Idunno;	
struct timeval startTime, endTime;

//	set a checkpoint and show the (natural) running time in seconds 
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time: %ld.%06ld s\n", sec_diff, usec_diff);
	printf("----------------------------------------------\n");
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


__global__
void runTimeStep(atom * atom_list, int n)
{
	//k.ashdglasdhg
}


void runMD(atom * atom_list, int n, float ts, float tf) { 

	int block_size = 32;

	// define memory requirements for atoms/histogram datasets.
	int atoms_size = n * sizeof(atom); 

        // write new device variable pointers
        atom *d_atom_list; // = atom_list;

	// allocate gpu memory and send data to gpu
        cudaMalloc((void**) &d_atom_list, atoms_size);
        cudaMemcpy(d_atom_list, atom_list, atoms_size, cudaMemcpyHostToDevice);

		dim3 dimGrid(ceil(n/block_size),1,1);
		dim3 dimBlock(block_size,1,1);

		//printf("%f\n",ceil(n/block_size));
	
		// time it
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
                	// go diego go
			for (float ti=0.0; ti <= tf; ti+=ts) {
                		runTimeStep<<< dimGrid, dimBlock >>>(d_atom_list, n);
                	}
		// fetch kernel runtime
		cudaEventRecord ( stop, 0 );
		cudaEventSynchronize( stop );
		float elapsedTime;
		cudaEventElapsedTime( &elapsedTime, start, stop );
		printf( "******** Total Running Time of doIt Kernel: %0.5f s ********\n", elapsedTime/1000.0 );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );

	
	// all done. Free up device memory.
        cudaFree(d_atom_list);
}

void readFile(System &system, atom * atom_list) {
        //printf("%le",system.constants.kb);
	string line;
        ifstream myfile ("test2.dat"); // test2.dat
        if (myfile.is_open())
        {
                //std::string::size_type sz;     // alias of size_t
                // loop through each line
                int id = 0;
                while ( getline (myfile,line) )
                {
			vector<string> lc;
                        istringstream iss(line);
                        //ostream_iterator<string> out_it (cout,",");
                        copy(
                                istream_iterator<string>(iss),
                                istream_iterator<string>(),
                                back_inserter(lc) // "normally" out_it goes here.
                        );

			// make the atom from the current line.
			atom current_atom;		

				
			current_atom.name[0] = lc[0].c_str()[0];
			current_atom.name[1] = lc[0].c_str()[1];
			current_atom.px = atof(lc[1].c_str());
			current_atom.py = atof(lc[2].c_str());
			current_atom.pz = atof(lc[3].c_str());
			current_atom.charge = atof(lc[4].c_str());
			current_atom.vx = 0.0;
			current_atom.vy = 0.0;
			current_atom.vz = 0.0;
			current_atom.ax = 0.0;
			current_atom.ay = 0.0;
			current_atom.az = 0.0;
			current_atom.fx = 0.0;
			current_atom.fy = 0.0;
			current_atom.fz = 0.0;
			current_atom.LJsig = system.constants.sigs[current_atom.name];
			current_atom.LJeps = system.constants.eps[current_atom.name];	
			
			printf("%c%c %f %f %f %f\n",current_atom.name[0],current_atom.name[1], current_atom.px, current_atom.py, current_atom.pz, current_atom.charge);
		
			atom_list[id] = current_atom;
			
			id++;
		}
	}
}


//// MAIN =============================================================
int main(int argc, char **argv)
{
	System system; 
	float ts = 1.0e-15;
	float tf = 1000e-15;
	
	// variable and memory assignments
	atom_list = (atom *)malloc(sizeof(atom)*75);
	//atom_list[200];

	// read da file
	readFile(system, atom_list);	
	
	// atom_list has been properly assigned
	//printf("%c%c %f",atom_list[0].name[0], atom_list[0].name[1], atom_list[0].px);

	int n =200 ; //atoi(argv[1]);

		//exit (EXIT_FAILURE);
		
		
	// time the entire GPU process. 
	gettimeofday(&startTime, &Idunno);

	// run the function which calls the kernel, times the kernel, etc.
        runMD(atom_list, n, ts, tf); // uses same atom list as cpu code

	// spit back runtime.
        report_running_time();
	
	return 0;
}

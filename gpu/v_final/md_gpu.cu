/* ==================================================================
	Programmers: Alfredo Peguero Tejada & Douglas Franz
	A molecular dynamics NVE code for GPU.
	To compile: nvcc my_file.cu -o my_exe in the rc machines
	run with, e.g. ./my_exe 
   ==================================================================
*/

// let's include our wonderful libraries
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
#include "constants.cpp" // this we made. Contains constants, i.e. force-field parameters and elemental masses, etc.
#include "system.cpp" // this we made. Contains class structure for chemical system management
#include <vector>

/* this is an explicit definition for atomicAdd, to be safe */
__device__ double atomicAdd(double* address, double val) 
{
 unsigned long long int* address_as_ull = (unsigned long long int*)address; 
  unsigned long long int old = *address_as_ull, assumed; 
  do { assumed = old; 
  old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
  } 
  while (assumed != old); 
  return __longlong_as_double(old); 
}


/* descriptors for single atom in the system */
typedef struct atomdesc {
	double px, py, pz; // positions
	double vx, vy, vz; // velocities
	double ax, ay, az; // accelerations 
	double fx, fy, fz; // forces 
	double charge, mass, LJsig, LJeps; // other params. LJ = Lennard-Jones
	char name[2];
} atom;

atom * atom_list; /* instance list of all data points  for GPU */

// These are for an old way of tracking time 
struct timezone Idunno;	
struct timeval startTime, endTime;

// set a checkpoint and show the (total CPU (host)) running time in seconds 
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

// force calculator, using LJ repulsion/dispersion and electrostatic force (i.e. Coloumb's law
__global__
void calculateForce(atom * atom_list, double ts, int n,double dimx,double dimy,double dimz)
{
	// an array for atoms in the shared set
	extern __shared__ atom sharedatom[];
	// define thread id
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// only run for real atoms (no ghost threads)
	if(i<n){	
		atom anchoratom=atom_list[i]; // make an anchor in register to save time.
	
		// distances etc.
		double dx,dy,dz,rsq,r,ux,uy,uz,fx,fy,fz,ke; // define variables
	
		// make sure we're synced across the block first.
		__syncthreads();


	//loop trough number of tiles = number of atoms/blocksize
	for(int tile=blockIdx.x;tile< (int) ceil( (double) n/blockDim.x);tile++){
	 	
		//if absolute atom index exists (may not be the case for last tile) load atom data
		if(threadIdx.x+tile*blockDim.x<n)	 
                	sharedatom[threadIdx.x]=atom_list[threadIdx.x+tile*blockDim.x];
                
		__syncthreads(); //make sure all atoms of tile are loaded
	 
	// loop through pairs
	//circular access of shared memory within tile
	int j=(threadIdx.x+1)%blockDim.x;
	do{
		int i2=tile*blockDim.x+j;//global index of neighboring atom
		if(i2<=i || i2>=n){
			j=(j+1)%blockDim.x;
			continue;
			}
		// ^^ that was the circular element ^^		 

	
		// apply mixing rules for LJ force
		double eps = sqrt(anchoratom.LJeps * sharedatom[j].LJeps);
		double sig = 0.5*(anchoratom.LJsig + sharedatom[j].LJsig);	
		
		// calculate distance
		dx = anchoratom.px - sharedatom[j].px;
		dy = anchoratom.py - sharedatom[j].py;
		dz = anchoratom.pz - sharedatom[j].pz;				
		
		// r = distance, and unit vectors
		rsq = dx*dx + dy*dy + dz*dz;
		r = sqrt(rsq);
		ux = dx/r;
		uy = dy/r;
		uz = dz/r;
		
		// calculate LJ force
		fx = 24*dx*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
		fy = 24*dy*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
		fz = 24*dz*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));

		// apply LJ forces	
		atomicAdd(&(atom_list[i].fx), fx);
		atomicAdd(&(atom_list[i].fy), fy);// += fy;
		atomicAdd(&(atom_list[i].fz), fz); // += fz;
			
		atomicAdd(&(atom_list[i2].fx), -fx);
                atomicAdd(&(atom_list[i2].fy), -fy);// += fy;
                atomicAdd(&(atom_list[i2].fz), -fz); // += fz;
			
		
		// calculat electrostatic force
		ke = 8.987551787e9; // Coloumb's constant
		fx = (ke * (anchoratom.charge * sharedatom[j].charge)/rsq) * ux;
                fy = (ke * (anchoratom.charge * sharedatom[j].charge)/rsq) * uy;
                fz = (ke * (anchoratom.charge * sharedatom[j].charge)/rsq) * uz;
			
		// apply electrostatic forces	
		atomicAdd(&(atom_list[i].fx), fx);
                atomicAdd(&(atom_list[i].fy), fy);// += fy;
                atomicAdd(&(atom_list[i].fz), fz); // += fz;

                atomicAdd(&(atom_list[i2].fx), -fx);
                atomicAdd(&(atom_list[i2].fy), -fy);// += fy;
                atomicAdd(&(atom_list[i2].fz), -fz); // += fz;
		
		//update atom index within shared memory
		j=(j+1)%blockDim.x;
		
			
		}while(j!=threadIdx.x);//go until you come back to starting index
		 __syncthreads();//make sure all forces are updated before loading more atoms

		}
	}
		
}

// after calculating force, this will be called to calculate new positions based on the calculated forces
__global__
void useTheForce(atom * atom_list, double ts, int n, double dimx, double dimy, double dimz) 
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n)
	{
		// assign atom to register
		atom anchoratom=atom_list[i];
	
		// variable definitions too
		double prev_ax,prev_ay,prev_az;
		prev_ax = anchoratom.ax;
		prev_ay = anchoratom.ay;
		prev_az = anchoratom.az;

		// a = F/m
		anchoratom.ax = anchoratom.fx / anchoratom.mass;
		anchoratom.ay = anchoratom.fy / anchoratom.mass;
		anchoratom.az = anchoratom.fz / anchoratom.mass;

		// new velocity (by Velocity Verlet algorithm: simply average of previous and next acc.)
		anchoratom.vx = anchoratom.vx + 0.5*(anchoratom.ax + prev_ax)*ts;
		anchoratom.vy = anchoratom.vy + 0.5*(anchoratom.ay + prev_ay)*ts;
		anchoratom.vz = anchoratom.vz + 0.5*(anchoratom.az + prev_az)*ts;
			
		// integrate to get position
		atom_list[i].px = anchoratom.px + anchoratom.vx * ts + 0.5 * anchoratom.ax * ts *ts;
		atom_list[i].py = anchoratom.py + anchoratom.vy * ts + 0.5 * anchoratom.ay * ts *ts;
		atom_list[i].pz = anchoratom.pz + anchoratom.vz * ts + 0.5 * anchoratom.az * ts *ts;
		
		// initialize forces just to be sure before calculating.	
		atom_list[i].fx = 0.0;
		atom_list[i].fy = 0.0;
        	atom_list[i].fz = 0.0;
	}
}

// an I/O function to output coordinate data for viewing videos etc.
void write(atom * atom_list, double time, int c, int n) 
{
	ofstream myfile;
	myfile.open ("outfile.xyz", ios_base::app);
	time = time * 1.0e15; // from fs to s
	myfile << n;
	myfile << "\n Time: ";
	myfile << time;
	myfile << " fs -- step count: ";
	myfile << c;
	myfile << "\n";

	for (int i =0; i < n; i++) {
		myfile << atom_list[i].name;  //"H ";
		myfile << "  ";
		myfile << atom_list[i].px*1e10;
		myfile << "  ";
		myfile << atom_list[i].py*1e10;
		myfile << "  ";
		myfile << atom_list[i].pz*1e10;
		myfile << "\n";
	} 
	myfile.close();
}

// the CPU function that runs the MD simulation by calling GPU kernels that modify atomic coordinates in a time- loop.
void runMD(atom * atom_list, int n, double ts, double tf,double dimx, double dimy,double dimz,int block_size) { 

	// define memory requirements for atoms/histogram datasets.
	int atoms_size = n * sizeof(atom); 

        // write new device variable pointers
        atom *d_atom_list; // = atom_list;

	// allocate gpu memory and send data to gpu to old
        cudaMalloc((void**) &d_atom_list, atoms_size);
        cudaMemcpy(d_atom_list, atom_list, atoms_size, cudaMemcpyHostToDevice);

		// grid and block sizes defined as 3D for future generalization
		dim3 dimGrid((int) ceil( (double) n/block_size),1,1);
		dim3 dimBlock(block_size,1,1);
	
		// time it
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
                	// RUN THE MD GPU KERNELS
			//int c = 0;
			for (double t=0.0; t <= tf; t+=ts) {
				// kernel functions for MD
                		useTheForce<<< dimGrid, dimBlock >>>( d_atom_list, ts, n,dimx,dimy,dimz );
				calculateForce<<< dimGrid, dimBlock,block_size*sizeof(atom) >>>( d_atom_list, ts, n,dimx,dimy,dimz);
				/*
				// OUTPUT TURNED OFF FOR TIME ANALYTICS TESTING	
				// extract data from GPU, write to output file, and print progress, as needed.
				if (c%1 == 0)
				{
					cudaMemcpy(atom_list, d_atom_list, atoms_size, cudaMemcpyDeviceToHost);
					write(atom_list, t, c, n);
					double prg = t/tf * 100;
					printf("Step #: %7i done;  Progress: %3.2f %%\n",c,prg);
				}
				c++;*/
                		
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

// a function to read input file data which is a list of elements, coordinates, and charges, e.g. H 0 1 2 0.14
void readFile(System &system, atom * atom_list,char* filename) {
	srand(time(NULL));
	string line;
        ifstream myfile (filename); // test2.dat
        if (myfile.is_open())
        {
                // loop through each line
                int id = 0;
                while ( getline (myfile,line) )
                {
			vector<string> lc;
                        istringstream iss(line);
                        copy(
                                istream_iterator<string>(iss),
                                istream_iterator<string>(),
                                back_inserter(lc)
                        );

			// make the atom in memory from the current line.
			atom ca;		

			ca.name[0] = lc[0].c_str()[0];
			ca.name[1] = lc[0].c_str()[1];
			ca.px = atof(lc[1].c_str())* system.constants.cA;
			ca.py = atof(lc[2].c_str())* system.constants.cA;
			ca.pz = atof(lc[3].c_str())* system.constants.cA;
			ca.charge = atof(lc[4].c_str())*system.constants.cC;
			
			// start with zero velocity for certain atoms
			if (ca.name[0]=='N' || ca.name[0]=='C' || ca.name[0]=='H' || ca.name[0]=='O'){
			ca.vx=0;
			ca.vy=0;
			ca.vz=0;
			}
			// else make a random initial velocity
			else{
			int maxV=10000;
			ca.vx = rand()%maxV-maxV/2;//0.0;
			ca.vy = rand()%maxV-maxV/2;//0.0;
			ca.vz = rand()%maxV-maxV/2;//0.0;
			}
			
			ca.ax = 0.0;
			ca.ay = 0.0;
			ca.az = 0.0;
			ca.fx = 0.0;
			ca.fy = 0.0;
			ca.fz = 0.0;
			ca.LJsig = system.constants.sigs[lc[0]];
			ca.LJeps = system.constants.eps[lc[0]];	
			ca.mass = system.constants.masses[lc[0]];

			atom_list[id] = ca;
			id++;
		}
	}
}


//// MAIN =============================================================
int main(int argc, char **argv)
{
	
	char* filename=argv[1]; // filename of input data
	
	// first delete outfile.xyz as needed.
	if ( remove( "outfile.xyz" ) != 0)
                perror( "Error deleting outfile.xyz" );
        else {
                cout << "outfile.xyz successfully deleted.";
                printf("\n");
        }

	int n = atoi(argv[2]); // number of atoms to simulate
	double dimx=atof(argv[3]); // box in x; irrelevant for NVE
	double dimy=atof(argv[4]); // box in y; irrelevant for NVE
	double dimz=atof(argv[5]); // box in z; irrelevant for NVE
	int block_size=atoi(argv[6]);	 // block size, e.g. 256 threads.
	System system; 
	double ts = 0.1e-15; // time step 1e-15 = 1 femptosecond.
	double tf = 1e-15; // final time to calculate.
	
	// variable and memory assignments
	atom_list = (atom *)malloc(sizeof(atom)*n);

	// read da file which assigns atoms to atom_list
	readFile(system, atom_list,filename);	

	// time the entire GPU process. 
	gettimeofday(&startTime, &Idunno);

	// run the function which calls the kernel, times the kernel, etc.
        runMD(atom_list, n, ts, tf,dimx,dimy,dimz,block_size); 

	// spit back runtime.
        report_running_time();
	
	return 0;
}



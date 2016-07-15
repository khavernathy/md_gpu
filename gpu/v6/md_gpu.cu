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
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) 
	{

		// calculate forces
		// initialize to zero.
		atom_list[i].fx = 0.0;
		atom_list[i].fy = 0.0;
		atom_list[i].fz = 0.0;

		// loop through pairs
		for (int j=i+1; j<n; j++) 
		{
			// check mixing rules
			double eps = sqrt(atom_list[i].LJeps * atom_list[j].LJeps);
			double sig = 0.5*(atom_list[i].LJsig + atom_list[j].LJsig);	
			
			// distances etc.
			double dx,dy,dz,rsq,r,ux,uy,uz,fx,fy,fz,ke;
			dx = atom_list[i].px - atom_list[j].px;
			dy = atom_list[i].py - atom_list[j].py;
			dz = atom_list[i].pz - atom_list[j].pz;				
				
			// r = distance, and unit vectors
			rsq = dx*dx + dy*dy + dz*dz;
			r = sqrt(rsq);
			ux = dx/r;
			uy = dy/r;
			uz = dz/r;

			// LJ force
			fx = 24*dx*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
			fy = 24*dy*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
			fz = 24*dz*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
				
			atomicAdd(&(atom_list[i].fx), fx);
			atomicAdd(&(atom_list[i].fy), fy);// += fy;
			atomicAdd(&(atom_list[i].fz), fz); // += fz;
			
			atomicAdd(&(atom_list[j].fx), -fx);
                        atomicAdd(&(atom_list[j].fy), -fy);// += fy;
                        atomicAdd(&(atom_list[j].fz), -fz); // += fz;

			// now electrostatic force
			ke = 8.987551787e9; // Coloumb's constant
			fx = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * ux;
                        fy = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * uy;
                        fz = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * uz;
					
			atomicAdd(&(atom_list[i].fx), fx);
                        atomicAdd(&(atom_list[i].fy), fy);// += fy;
                        atomicAdd(&(atom_list[i].fz), fz); // += fz;

                        atomicAdd(&(atom_list[j].fx), -fx);
                        atomicAdd(&(atom_list[j].fy), -fy);// += fy;
                        atomicAdd(&(atom_list[j].fz), -fz); // += fz;

			/*
			// now we check if a periodic-boundary-condition phantom atom is required ( to better simulate a macroscopic system )	
			bool phantom=false;
			if(dx>0.5*dimx){
				dx-=dimx;
				phantom=true;
			}
			else if(dx<=-0.5*dimx){	
				dx+=dimx;
				phantom=true;
			}
			if(dy>0.5*dimy){
				dy-=dimy;
				phantom=true;
			}
			else if(dy<=-0.5*dimy){
				dy+=dimy;
				phantom=true;
			}
			if(dz>0.5*dimz){
				dz-=dimz;
				phantom=true;
			}
			else if(dz<=-0.5*dimz){
				dz+=dimz;
				phantom=true;
			}
			
					
			// now add force contributions of phantom atom if it is needed for PBC	
			if(phantom){
                                rsq = dx*dx + dy*dy + dz*dz;
                                r = sqrt(rsq);
                                ux = dx/r;
                                uy = dy/r;
                                uz = dz/r;

                                // LJ force
                                fx = 24*dx*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
                                fy = 24*dy*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
                                fz = 24*dz*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));

                                atomicAdd(&(atom_list[i].fx), fx);
                                atomicAdd(&(atom_list[i].fy), fy);// += fy;
                                atomicAdd(&(atom_list[i].fz), fz); // += fz;

                                atomicAdd(&(atom_list[j].fx), -fx);
                                atomicAdd(&(atom_list[j].fy), -fy);// += fy;
                                atomicAdd(&(atom_list[j].fz), -fz); // += fz;
                                
				 // electrostatic force
                                ke = 8.987551787e9;
                                fx = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * ux;
                                fy = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * uy;
                                fz = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * uz;

                                atomicAdd(&(atom_list[i].fx), fx);
                                atomicAdd(&(atom_list[i].fy), fy);// += fy;
                                atomicAdd(&(atom_list[i].fz), fz); // += fz;

                                atomicAdd(&(atom_list[j].fx), -fx);
                                atomicAdd(&(atom_list[j].fy), -fy);// += fy;
                                atomicAdd(&(atom_list[j].fz), -fz); // += fz;

			}*/
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
		double prev_ax,prev_ay,prev_az;
		prev_ax = atom_list[i].ax;
		prev_ay = atom_list[i].ay;
		prev_az = atom_list[i].az;

		// a = F/m
		atom_list[i].ax = atom_list[i].fx / atom_list[i].mass;
		atom_list[i].ay = atom_list[i].fy / atom_list[i].mass;
		atom_list[i].az = atom_list[i].fz / atom_list[i].mass;

		// new velocity (by Velocity Verlet algorithm)
		atom_list[i].vx = atom_list[i].vx + 0.5*(atom_list[i].ax + prev_ax)*ts;
		atom_list[i].vy = atom_list[i].vy + 0.5*(atom_list[i].ay + prev_ay)*ts;
		atom_list[i].vz = atom_list[i].vz + 0.5*(atom_list[i].az + prev_az)*ts;
			
		// integrate to get position
		atom_list[i].px = atom_list[i].px + atom_list[i].vx * ts + 0.5 * atom_list[i].ax * ts *ts;
		atom_list[i].py = atom_list[i].py + atom_list[i].vy * ts + 0.5 * atom_list[i].ay * ts *ts;
		atom_list[i].pz = atom_list[i].pz + atom_list[i].vz * ts + 0.5 * atom_list[i].az * ts *ts;
		
		/*
		// finally, out of the box check. If atom has escaped the box, move it back over by one box length
		if ( atom_list[i].px * 1e10 >= dimx/2.0 ) 
			atom_list[i].px -= (dimx * 1e-10);
		if ( atom_list[i].px * 1e10 <= -dimx/2.0 )
			atom_list[i].px += (dimx * 1e-10);

		if ( atom_list[i].py * 1e10 >= dimy/2.0 )
			atom_list[i].py -= (dimy * 1e-10);
		if ( atom_list[i].py * 1e10 <= -dimy/2.0 )
			atom_list[i].py += (dimy * 1e-10);
		
		if ( atom_list[i].pz * 1e10 >= dimz/2.0 )
			atom_list[i].pz -= (dimz * 1e-10);
		if ( atom_list[i].pz * 1e10 <= -dimz/2.0 )
			atom_list[i].pz += (dimz * 1e-10);
		*/
		//printf("%le\n",atom_list[i].px);
		
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
void runMD(atom * atom_list, int n, float ts, float tf,double dimx, double dimy,double dimz) { 

	// our old friend block-size
	int block_size = 32; // # of threads

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
                	// go diego go
			int c = 0;
			for (float t=0.0; t <= tf; t+=ts) {
				// kernel functions for MD
                		useTheForce<<< dimGrid, dimBlock >>>( d_atom_list, ts, n,dimx,dimy,dimz );
				calculateForce<<< dimGrid, dimBlock >>>( d_atom_list, ts, n,dimx,dimy,dimz );
					
				// extract data from GPU, write to output file, and print progress, as needed.
				if (c%10 == 0)
				{
					cudaMemcpy(atom_list, d_atom_list, atoms_size, cudaMemcpyDeviceToHost);
					write(atom_list, t, c, n);
					double prg = t/tf * 100;
					printf("Step #: %7i done;  Progress: %3.2f %%\n",c,prg);
				}
				c++; 
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
        cudaFree(d_atom_list); //cudaFree(d_atom_list);
}

// a function to read input file data which is a list of elements, coordinates, and charges, e.g. H 0 1 2 0.14
void readFile(System &system, atom * atom_list,char* filename) {
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
                                back_inserter(lc) // "normally" out_it goes here.
                        );

			// make the atom in memory from the current line.
			atom ca;		

			ca.name[0] = lc[0].c_str()[0];
			ca.name[1] = lc[0].c_str()[1];
			ca.px = atof(lc[1].c_str())* system.constants.cA;
			ca.py = atof(lc[2].c_str())* system.constants.cA;
			ca.pz = atof(lc[3].c_str())* system.constants.cA;
			ca.charge = atof(lc[4].c_str())*system.constants.cC;
			ca.vx = 0.0;
			ca.vy = 0.0;
			ca.vz = 0.0;
			ca.ax = 0.0;
			ca.ay = 0.0;
			ca.az = 0.0;
			ca.fx = 0.0;
			ca.fy = 0.0;
			ca.fz = 0.0;
			ca.LJsig = system.constants.sigs[lc[0]];
			ca.LJeps = system.constants.eps[lc[0]];	
			ca.mass = system.constants.masses[lc[0]];

			//printf("%c%c %le %le %le %f %f %f %f %f %f %f %f %f %f %le %le %le\n",ca.name[0],ca.name[1], ca.px, ca.py, ca.pz, ca.charge, ca.vx, ca.vy, ca.vz, ca.ax, ca.ay, ca.az, ca.fx, ca.fy, ca.fz, ca.LJsig, ca.LJeps, ca.mass);		
			atom_list[id] = ca;
			id++;
		}
	}
}


//// MAIN =============================================================
int main(int argc, char **argv)
{
	printf("%i",sizeof(atom));

	char* filename=argv[1];
	// first delete outfile.xyz as needed.
	if ( remove( "outfile.xyz" ) != 0)
                perror( "Error deleting outfile.xyz" );
        else {
                cout << "outfile.xyz successfully deleted.";
                printf("\n");
        }

	int n = atoi(argv[2]);
	double dimx=atof(argv[3]); // for our water system, these 3 are 18
	double dimy=atof(argv[4]);
	double dimz=atof(argv[5]);
	
	System system; 
	float ts = 0.1e-15;
	float tf = 100e-15;
	
	// variable and memory assignments
	atom_list = (atom *)malloc(sizeof(atom)*n);

	// read da file which assigns atoms to atom_list
	readFile(system, atom_list,filename);	

	// time the entire GPU process. 
	gettimeofday(&startTime, &Idunno);

	// run the function which calls the kernel, times the kernel, etc.
        runMD(atom_list, n, ts, tf,dimx,dimy,dimz); // uses same atom list as cpu code

	// spit back runtime.
        report_running_time();

	return 0;
}

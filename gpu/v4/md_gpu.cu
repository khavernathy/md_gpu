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
/*
__global__
void newToOld(atom * new_list, atom * old_list, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n)
	{
		old_list[i] = new_list[i];
	}
}
*/

__global__
void calculateForce(atom * atom_list, double ts, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) 
	{
		//printf("%le\n",atom_list[i].pz); //works

				// calculate forces
		// initialize to zero.
		atom_list[i].fx = 0.0;
		atom_list[i].fy = 0.0;
		atom_list[i].fz = 0.0;

		// loop through pairs
		for (int j=i+1; j<n; j++) 
		{
				//initialize atom j force = 0
				//atom_list[j].fx = 0.0;	
				//atom_list[j].fy = 0.0;
				//atom_list[j].fz = 0.0;
				
				// check mixing rules
				double eps = sqrt(atom_list[i].LJeps * atom_list[j].LJeps);
				double sig = 0.5*(atom_list[i].LJsig + atom_list[j].LJsig);

				// distances etc.
				double dx,dy,dz,rsq,r,ux,uy,uz,fx,fy,fz,ke;
				dx = atom_list[i].px - atom_list[j].px;
				dy = atom_list[i].py - atom_list[j].py;
				dz = atom_list[i].pz - atom_list[j].pz;

				rsq = dx*dx + dy*dy + dz*dz;
				r = sqrt(rsq);
				ux = dx/r;
				uy = dy/r;
				uz = dz/r;

				// LJ force
				fx = 24*dx*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
				fy = 24*dy*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
				fz = 24*dz*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6)*pow(r,-8));
				
				atom_list[i].fx += fx;
				atom_list[i].fy += fy;// += fy;
				atom_list[i].fz +=fz; // += fz;
			
				atom_list[j].fx += fx; //atom_list[j].fx -= fx;
				atom_list[j].fy += fy; // -= fy;
				atom_list[j].fz += fz; // -= fz;	
				
				// electrostatic force
				ke = 8.987551787e9;
				fx = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * ux;
                                fy = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * uy;
                                fz = (ke * (atom_list[i].charge * atom_list[j].charge)/rsq) * uz;

                                atom_list[i].fx +=fx; // += fx;
                                atom_list[i].fy+= fy; // += fy;
                                atom_list[i].fz += fz; // += fz;

                                atom_list[j].fx+= fx; // -= fx;
                                atom_list[j].fy+= fy; // -= fy;
                                atom_list[j].fz+= fz; // -= fz;

		}


	}	
}

__global__
void useTheForce(atom * atom_list, double ts, int n) 
{
	// do velocity verlet on forces to get accel. / vel.
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

		// new velocity
		atom_list[i].vx = atom_list[i].vx + 0.5*(atom_list[i].ax + prev_ax)*ts;
		atom_list[i].vy = atom_list[i].vy + 0.5*(atom_list[i].ay + prev_ay)*ts;
		atom_list[i].vz = atom_list[i].vz + 0.5*(atom_list[i].az + prev_az)*ts;
			
		// integrate
		//integrate(atom_list[i], atom_list[i], ts);			
		atom_list[i].px = atom_list[i].px + atom_list[i].vx * ts + 0.5 * atom_list[i].ax * ts *ts;
		atom_list[i].py = atom_list[i].py + atom_list[i].vy * ts + 0.5 * atom_list[i].ay * ts *ts;
		atom_list[i].pz = atom_list[i].pz + atom_list[i].vz * ts + 0.5 * atom_list[i].az * ts *ts;
		

	}
}

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
		//myfile << atom_list[i].name;//[0];
		//myfile << atom_list[i].name[1]; 
		//printf("%s",atom_list[i].name);
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

void runMD(atom * atom_list, int n, float ts, float tf) { 

	//printf("%s",atom_list[72].name);
	//printf("%le",atom_list[72].pz);

	int block_size = 32;

	// define memory requirements for atoms/histogram datasets.
	int atoms_size = n * sizeof(atom); 

        // write new device variable pointers
        atom *d_atom_list; // = atom_list;
	//atom *d_atom_list; 

	// allocate gpu memory and send data to gpu to old
        cudaMalloc((void**) &d_atom_list, atoms_size);
        cudaMemcpy(d_atom_list, atom_list, atoms_size, cudaMemcpyHostToDevice);

	// and the new (duplicate)
	//cudaMalloc((void**) &d_atom_list, atoms_size);
	//cudaMemcpy(d_atom_list, atom_list, atoms_size, cudaMemcpyHostToDevice);
	
		dim3 dimGrid(ceil(n/block_size),1,1);
		dim3 dimBlock(block_size,1,1);
	
		// time it
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
                	// go diego go
			int c = 0;
			for (float t=0.0; t <= tf; t+=ts) {
                		useTheForce<<< dimGrid, dimBlock >>>( d_atom_list, ts, n );
				calculateForce<<< dimGrid, dimBlock >>>( d_atom_list, ts, n );
				//newToOld<<< dimGrid, dimBlock >>>( d_atom_list, d_atom_list, n );
					
				// write to output file as needed.
				if (c%1 == 0)
				{
					// old and new are same at this point, so whatevs
					cudaMemcpy(atom_list, d_atom_list, atoms_size, cudaMemcpyDeviceToHost);
					//printf("%s",atom_list[0].px);
					write(atom_list, t, c, n);
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
			atom ca;		

			ca.name[0] = lc[0].c_str()[0];
			ca.name[1] = lc[0].c_str()[1];
			ca.px = atof(lc[1].c_str())* system.constants.cA;
			ca.py = atof(lc[2].c_str())* system.constants.cA;
			ca.pz = atof(lc[3].c_str())* system.constants.cA;
			ca.charge = atof(lc[4].c_str());
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

	// first delete outfile.xyz as needed.
	if ( remove( "outfile.xyz" ) != 0)
                perror( "Error deleting outfile.xyz" );
        else {
                cout << "outfile.xyz successfully deleted.";
                printf("\n");
        }

	int n = 75;
	
	System system; 
	float ts = 1.0e-15;
	float tf = 100e-15;
	
	// variable and memory assignments
	atom_list = (atom *)malloc(sizeof(atom)*n);
	//atom_list[200];

	// read da file which assigns atoms to atom_list
	readFile(system, atom_list);	

	//write(atom_list, 0, 0, n);
	
	//printf("%s",atom_list[0].name);
	
	// time the entire GPU process. 
	gettimeofday(&startTime, &Idunno);

	// run the function which calls the kernel, times the kernel, etc.
        runMD(atom_list, n, ts, tf); // uses same atom list as cpu code

	// spit back runtime.
        report_running_time();
	
	return 0;
}

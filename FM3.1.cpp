#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <map>
#include <string>
#include <constants.cpp>
#include <system.cpp>
#include <vector>
//#include <boost/foreach.hpp>
//#include <boost/tokenizer.hpp>

using namespace std;

void readInAtoms(System &system) {
	string line;
	ifstream myfile ("test2.dat"); // test2.dat
	if (myfile.is_open())
	{
		std::string::size_type sz;     // alias of size_t
		// loop through each line
		int id = 1;
		while ( getline (myfile,line) )
		{
			vector<string> myvector;
      			istringstream iss(line);
			//ostream_iterator<string> out_it (cout,",");	
			copy(
				istream_iterator<string>(iss),
				istream_iterator<string>(),
				back_inserter(myvector) // "normally" out_it goes here.
			);
	

			//temporary class instance current_atom
			Atom current_atom;
			
			current_atom.name = myvector[0];
			current_atom.m = system.constants.masses[current_atom.name];
			current_atom.eps = system.constants.eps[current_atom.name];
			current_atom.sig = system.constants.sigs[current_atom.name];
			current_atom.V = 0.0;
			current_atom.K = 0.0;
			current_atom.E = 0.0;
			current_atom.ID = id;
			current_atom.pos["x"] = stod(myvector[1]) * system.constants.cA; // convert distance (Angstroems) in input file to meters.
			current_atom.pos["y"] = stod(myvector[2]) * system.constants.cA;
			current_atom.pos["z"] = stod(myvector[3]) * system.constants.cA;
			current_atom.force["x"] = 0.0;
			current_atom.force["y"] = 0.0;
			current_atom.force["z"] = 0.0;
			current_atom.vel["x"] = 0.0;
			current_atom.vel["y"] = 0.0;
			current_atom.vel["z"] = 0.0;
			current_atom.acc["x"] = 0.0;
			current_atom.acc["y"] = 0.0;
			current_atom.acc["z"] = 0.0;
			current_atom.C = stod(myvector[4]) * system.constants.cC; // convert charge (e-) in input file to Coulombs.
			system.atoms.push_back(current_atom);
			//cout << '\n';
			//cout << system.atoms[id].name << ' ' << system.atoms[id].pos["x"] << ' ' << system.atoms[id].pos["y"] << ' ' << system.atoms[id].pos["z"] << ' ' << system.atoms[id].C;
			//cout << '\n' <<  system.atoms[id].m;
			//cout << '\n';
			id++;
			// this would print the whole line.
			//cout << line << '\n';
		}
		//printf("%i\n", system.atoms.size());
		myfile.close();
		
	}
	else cout << "Unable to open file"; 
}


void calculateForces(System &system) {
	// loop through all atoms
	long unsigned int size = system.atoms.size();
	for (int i = 0; i < size; i++) {
		//printf("%f\n",system.atoms[i].pos["z"]);
		// initialize force and potential to zero.
		system.atoms[i].force["x"] = 0.0;
		system.atoms[i].force["y"] = 0.0;
		system.atoms[i].force["z"] = 0.0;
		system.atoms[i].V = 0.0;
	}
	
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			// only do half matrix calculation
			if (system.atoms[i].ID < system.atoms[j].ID) {
				
				// check for mixing rules
				double eps,sig;
				if (system.atoms[i].eps != system.atoms[j].eps) {
					eps = sqrt(system.atoms[i].eps * system.atoms[j].eps);
				} else {
					eps = system.atoms[i].eps;
				}
				if (system.atoms[i].sig != system.atoms[j].sig) {
					sig = 0.5*(system.atoms[i].sig + system.atoms[j].sig);
				} else {
					sig = system.atoms[i].sig;
				}

				// preliminary calculations (distance between atoms, etc.)
				double dx,dy,dz,rsq,r,ux,uy,uz;
				dx = system.atoms[i].pos["x"] - system.atoms[j].pos["x"];
				dy = system.atoms[i].pos["y"] - system.atoms[j].pos["y"];
				dz = system.atoms[i].pos["z"] - system.atoms[j].pos["z"];
					
				rsq = dx*dx + dy*dy + dz*dz;
				r = sqrt(rsq);
				ux = dx/r;
				uy = dy/r;
				uz = dz/r;
				
				// Lennard-Jones force calculations
				double fx,fy,fz;
				fx = 24.0*dx*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6) * pow(r,-8));
				fy = 24.0*dy*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6) * pow(r,-8));
				fz = 24.0*dz*eps*(2*pow(sig,12)*pow(r,-14) - pow(sig,6) * pow(r,-8));
				
				system.atoms[i].force["x"] += fx;
				system.atoms[i].force["y"] += fy;
				system.atoms[i].force["z"] += fz;

				system.atoms[j].force["x"] -= fx;
				system.atoms[j].force["y"] -= fy;
				system.atoms[j].force["z"] -= fz;

				// LJ Potential
				system.atoms[i].V += 4.0*eps*(pow(sig/r,12) - pow(sig/r,6));


				
				// Coulomb's law electrostatic force. Overwrite fx,fy,fz
				fx = (system.constants.ke * (system.atoms[i].C * system.atoms[j].C)/rsq) * ux;
				fy = (system.constants.ke * (system.atoms[i].C * system.atoms[j].C)/rsq) * uy;
				fz = (system.constants.ke * (system.atoms[i].C * system.atoms[j].C)/rsq) * uz;	

				system.atoms[i].force["x"] += fx;
				system.atoms[i].force["y"] += fy;
                                system.atoms[i].force["z"] += fz;

                                system.atoms[j].force["x"] -= fx;
                                system.atoms[j].force["y"] -= fy;
                                system.atoms[j].force["z"] -= fz;
			
				//Coulombic potential
				system.atoms[i].V += (system.constants.ke*(system.atoms[i].C * system.atoms[j].C)/r);
				
			}
		}
	}
}

void integrate(System &system, double dt) {
	// Use velocity verlet method.
	long unsigned int size = system.atoms.size();
        for (int i = 0; i < size; i++) {
		// first get new position.
		system.atoms[i].pos["x"] = system.atoms[i].pos["x"] + system.atoms[i].vel["x"] * dt + 0.5*system.atoms[i].acc["x"] * dt * dt; // * pow(dt,2); FM3.1 update
		system.atoms[i].pos["y"] = system.atoms[i].pos["y"] + system.atoms[i].vel["y"] * dt + 0.5*system.atoms[i].acc["y"] * dt * dt; // * pow(dt,2);
		system.atoms[i].pos["z"] = system.atoms[i].pos["z"] + system.atoms[i].vel["z"] * dt + 0.5*system.atoms[i].acc["z"] * dt * dt; // * pow(dt,2);	
	}
	// get new forces for the new position.
	calculateForces(system);
	// loop through and get new acceleration and velocity.
	for (int i = 0; i < size; i++) {
        	// temporary current accelerations.
		double prev_ax,prev_ay,prev_az;
		prev_ax = system.atoms[i].acc["x"];
		prev_ay = system.atoms[i].acc["y"];
		prev_az = system.atoms[i].acc["z"];
		
		// a = F/m	
		system.atoms[i].acc["x"] = system.atoms[i].force["x"] / system.atoms[i].m;
		system.atoms[i].acc["y"] = system.atoms[i].force["y"] / system.atoms[i].m;
		system.atoms[i].acc["z"] = system.atoms[i].force["z"] / system.atoms[i].m;
		
		// new velocity
		system.atoms[i].vel["x"] = system.atoms[i].vel["x"] + 0.5*(system.atoms[i].acc["x"] + prev_ax)*dt;
		system.atoms[i].vel["y"] = system.atoms[i].vel["y"] + 0.5*(system.atoms[i].acc["y"] + prev_ay)*dt;
		system.atoms[i].vel["z"] = system.atoms[i].vel["z"] + 0.5*(system.atoms[i].acc["z"] + prev_az)*dt;
	}

}

double * calculateEnergyAndTemp(System &system) { // the * is to return an array of doubles as a pointer, not just one double
	double E_total = 0.0;
	double K_total = 0.0;
	double T=0.0;
	long unsigned int size = system.atoms.size();
        for (int i = 0; i < size; i++) {
		// v^2 is the dot product of the velocity vector with itself, square rooted, squared.
		double vsquared = pow(sqrt(system.atoms[i].vel["x"] * system.atoms[i].vel["x"] + system.atoms[i].vel["y"] * system.atoms[i].vel["y"] + system.atoms[i].vel["z"] * system.atoms[i].vel["z"]),2);
		
		// calculate kinetic
		system.atoms[i].K = 0.5*system.atoms[i].m * vsquared;
		// iteratively sum kinetic
		K_total += system.atoms[i].K;

		//total energy
		system.atoms[i].E = system.atoms[i].V + system.atoms[i].K;
		// iteratively sum total energy
		E_total += system.atoms[i].E;
	}

	// calculate temperature from kinetic energy and number of particles
	int dof = 6; // degrees of freedom. Not sure what correct value is.
	T = K_total / ((3.0 * system.atoms.size() - dof ) * system.constants.kb);
	//printf("Temperature: %4.2fK ",T); 

	static double output[2];
	output[0] = E_total;
	output[1] = T;
	return output;
}

void write(System &system, double time, int c) {
	ofstream myfile;
	myfile.open ("outfile.xyz", ios_base::app);
	long unsigned int size = system.atoms.size();
	time = time * 1.0e15; // time expressed in fs
 	myfile << to_string(size) + "\n Time: " + to_string(time) + " fs -- step count: " + to_string(c) + "\n";

	for (int i = 0; i < size; i++) {
		myfile << system.atoms[i].name + "   " + to_string(system.atoms[i].pos["x"]*1e10) + "   " + to_string(system.atoms[i].pos["y"]*1e10) + "   " + to_string(system.atoms[i].pos["z"]*1e10) + "\n";
	}


  	myfile.close();
}

int main() {
	//Constants constants;
	//write();
	System system;
	readInAtoms(system);
	// remove outfile.xyz if it exists
	if ( remove( "outfile.xyz" ) != 0)
		perror( "Error deleting outfile.xyz" );
	else {
		cout << "outfile.xyz successfully deleted.";
		printf("\n");
	}
	
	// begin writing new outfile.
	write(system,0,0);
	
	double dt = 0.2e-15; // 1e-15 is one femptosecond.
	double tf = 10000e-15; // 100,000e-15 would be 1e-9 seconds, or 1 nanosecond. 
	int c = 1;
	for (double t=0; t < tf; t=t+dt) {
		integrate(system,dt);
		double* ETarray = new double[2];		
		ETarray = calculateEnergyAndTemp(system);
		write(system,(t+dt),c);
		double prg = t/tf * 100;
		printf("Step #: %7i done; Progress: %3.2f %%; Energy: %10.10e J; Temperature: %4.2fK = %4.2f\370C\n",c,prg,ETarray[0],ETarray[1],ETarray[1]-273.15);
		c++;
	}

	//printf("%i\n",system.atoms.size());
	//printf("Atom # 14: %20.10e\n",::system.atoms[14].m);
	
	//printf("Mass of H: %20.10e\n",constants.masses["H"]);
	//printf("kb: %e\n",constants.kb);
	//printf("ke: %e\n",constants.ke);
	//printf("Eps of Cl: %20.10e\n",constants.eps["Cl"]);
	
	return(0);

}

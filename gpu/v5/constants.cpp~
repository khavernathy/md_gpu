#include <stdio.h>
#include <math.h>
#include <string>
#include <map>
// this is a test

using namespace std;

class Constants {
	public:
		Constants();
		double kb,fs,cC,ke,eV,cM,cA,cJ;
		map <string,double> masses;
		map <string,double> sigs;
		map <string,double> eps;
};

class Atom {
	public:
		Atom();
		string name;
		double m,eps,sig,C,V,K,E;
		int ID;
		map <string,double> pos; // equivalent of python FM1.9 "x"
		map <string,double> force; // "    " F
		map <string,double> vel; // v
		map <string,double> acc; // a
		
};

Atom::Atom() {}

Constants::Constants() {
	kb = 1.3806488e-23; // Boltzmann's in J/K
	fs = 1.0e-15; // seconds in a femptosecond
	cC = 1.60217662e-19; // cC coulombs = 1e
	ke = 8.987551787e9; // ke, Coulomb's constant, Nm^2/C^2
	eV = 6.242e18; // 1J = eV electron volts
	cM = 1.660578e-27; // kg / particle from g/mol
	cA = 1.0e-10; // 1 angstroem = cA meters
	cJ = 6.94786e-21; // 1 kcal/mol = cJ Joules

	masses["H"] = 1.008*cM;
	masses["He"] = 4.002602*cM;
	masses["Li"] = 6.941*cM;
	masses["Be"] = 9.012182*cM;
	masses["B"] = 10.811*cM;
	masses["Ne"] = 20.1797*cM;
	masses["Xe"] = 131.293*cM;
	masses["C"] = 12.011*cM;
	masses["N"] = 14.007*cM;
	masses["O"] = 15.9998*cM;
	masses["F"] = 18.998*cM;
	masses["Si"] = 28.085*cM;
	masses["P"] = 30.973*cM;
	masses["S"] = 32.06*cM;
	masses["Cl"] = 35.45*cM;
	masses["Na"] = 22.98976928*cM;
	masses["Zn"] = 65.39*cM;

	sigs["H"] = 0.5*cA; // Rappe=2.886; I changed to 0.5 for water model; 0.3 for MOF5.
	sigs["He"] = 2.362*cA;
	sigs["Li"] = 2.451*cA;
	sigs["Be"] = 2.745*cA;
	sigs["B"] = 4.083*cA;
	sigs["N"] = 3.66*cA;
	sigs["O"] = 1.5*cA; // Rappe=3.5; I changed to 1.5 for water model (SPC = 3.166); 1.3 for MOF5
	sigs["F"] = 3.364*cA;
	sigs["Ne"] = 3.243*cA;
	sigs["Xe"] = 4.404*cA;
	sigs["C"] = 3.851*cA; // Rappe = 3.851; I changed to 1.3 for MOF5
	sigs["Cl"] = 3.947*cA;
	sigs["Na"] = 2.983*cA;
	sigs["Zn"] = 2.763*cA;

	eps["H"] = 0.044*cJ;
	eps["He"] = 0.056*cJ;
	eps["Li"] = 0.025*cJ;
	eps["Be"] = 0.085*cJ;
	eps["B"] = 0.18*cJ;
	eps["N"] = 0.069*cJ;
	eps["O"] = 0.06*cJ;
	eps["F"] = 0.05*cJ;
	eps["Ne"] = 0.042*cJ;
	eps["Xe"] = 0.332*cJ;
	eps["C"] = 0.105*cJ;
	eps["Cl"] = 0.227*cJ;
	eps["Na"] = 0.03*cJ;
	eps["Zn"] = 0.124*cJ;
}

#include <stdio.h>
#include <math.h>
#include <string>
#include <map>
#include <vector>

using namespace std;

class System {
	public:
		System();
		vector<Atom> atoms;
		Constants constants;
};

System::System() {
}

#include "geweke_tester.hpp"

int main(){

	baxcat::GewekeTester gwk(10,1,35515);

    size_t lag = 5;

	gwk.run(100000,5,lag);

	gwk.outputResults();

	return 0;
}
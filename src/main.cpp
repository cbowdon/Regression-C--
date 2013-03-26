#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "Regression.hpp"

using namespace ml;
using namespace std;

void test_lin ()
{
	LinearRegression machine;

	Matd features = (Matd(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	Matd targets = (Matd(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

	machine.train(features, targets);

	cout << "Params:\n" << machine.get_params() << endl;
}

void test_lin2 ()
{
	LinearRegression machine;

	Matd features = (Matd(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	Matd targets = (Matd(10, 1) << -2, -4, -6, -8, -10, -12, -14, -16, -18, -20);

	machine.train(features, targets);

	cout << "Params:\n" << machine.get_params() << endl;
}

void test_1d ()
{
	LogisticRegression machine;

	Matd features = (Matd(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	Matd targets = (Matd(10, 1) <<  0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

	Matd test0 = (Matd(1, 1) << 1);
	Matd test1 = (Matd(1, 1) << 2);
	Matd test2 = (Matd(1, 1) << 3);
	Matd test3 = (Matd(1, 1) << 4);
	Matd test4 = (Matd(1, 1) << 5);
	Matd test5 = (Matd(1, 1) << 6);
	Matd test6 = (Matd(1, 1) << 7);
	Matd test7 = (Matd(1, 1) << 8);
	Matd test8 = (Matd(1, 1) << 9);
	Matd test9 = (Matd(1, 1) << 10);

	machine.train(features, targets);

	cout << "Params:\n" << machine.get_params() << endl;

	cout << "Predictions:\n" << endl;
	cout << "1:\t" << machine.predict(test0) << endl;
	cout << "2:\t" << machine.predict(test1) << endl;
	cout << "3:\t" << machine.predict(test2) << endl;
	cout << "4:\t" << machine.predict(test3) << endl;
	cout << "5:\t" << machine.predict(test4) << endl;
	cout << "6:\t" << machine.predict(test5) << endl;
	cout << "7:\t" << machine.predict(test6) << endl;
	cout << "8:\t" << machine.predict(test7) << endl;
	cout << "9:\t" << machine.predict(test8) << endl;
	cout << "10:\t" << machine.predict(test9) << endl;
}

void test_or ()
{
	LogisticRegression machine;

	// Test case: OR
	Matd features = (Matd(4, 2) << 
			0, 0, 
			0, 1, 
			1, 0, 
			1, 1);
	Matd targets = (Matd(4, 1) << 0, 1, 1, 1);

	Matd test0 = (Matd(1, 2) << 0, 0); // -> 0
	Matd test1 = (Matd(1, 2) << 0, 1); // -> 1
	Matd test2 = (Matd(1, 2) << 1, 0); // -> 1
	Matd test3 = (Matd(1, 2) << 1, 1); // -> 1

	cout << "Features:\n" << features << endl;
	cout << "Targets:\n" << targets << endl;

	machine.train(features, targets);

	cout << "Params:\n" << machine.get_params() << endl;

	cout << "Predictions:\n" << endl;
	cout << "00:\t" << machine.predict(test0) << endl;
	cout << "01:\t" << machine.predict(test1) << endl;
	cout << "10:\t" << machine.predict(test2) << endl;
	cout << "11:\t" << machine.predict(test3) << endl;
}

int main (int argc, char** argv)
{
	test_lin();
	test_or();

	return EXIT_SUCCESS;
}

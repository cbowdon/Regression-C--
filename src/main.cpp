#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "LogisticRegression.hpp"

using namespace ml;
using namespace std;

void test_lin ()
{
	LogisticRegression machine;

	Mat2d features = (Mat2d(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	Mat2d targets = (Mat2d(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

	machine.train(features, targets);

	cout << "Params:\n" << machine.get_params() << endl;
}

void test_lin2 ()
{
	LogisticRegression machine;

	Mat2d features = (Mat2d(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	Mat2d targets = (Mat2d(10, 1) << -2, -4, -6, -8, -10, -12, -14, -16, -18, -20);

	machine.train(features, targets);

	cout << "Params:\n" << machine.get_params() << endl;
}

void test_1d ()
{
	LogisticRegression machine;

	Mat2d features = (Mat2d(10, 1) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	Mat2d targets = (Mat2d(10, 1) <<  0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

	Mat2d test0 = (Mat2d(1, 1) << 1);
	Mat2d test1 = (Mat2d(1, 1) << 2);
	Mat2d test2 = (Mat2d(1, 1) << 3);
	Mat2d test3 = (Mat2d(1, 1) << 4);
	Mat2d test4 = (Mat2d(1, 1) << 5);
	Mat2d test5 = (Mat2d(1, 1) << 6);
	Mat2d test6 = (Mat2d(1, 1) << 7);
	Mat2d test7 = (Mat2d(1, 1) << 8);
	Mat2d test8 = (Mat2d(1, 1) << 9);
	Mat2d test9 = (Mat2d(1, 1) << 10);

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
	Mat2d features = (Mat2d(4, 2) << 
			0, 0, 
			0, 1, 
			1, 0, 
			1, 1);
	Mat2d targets = (Mat2d(4, 1) << 0, 1, 1, 1);

	Mat2d test0 = (Mat2d(1, 2) << 0, 0); // -> 0
	Mat2d test1 = (Mat2d(1, 2) << 0, 1); // -> 1
	Mat2d test2 = (Mat2d(1, 2) << 1, 0); // -> 1
	Mat2d test3 = (Mat2d(1, 2) << 1, 1); // -> 1

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
	test_or();

	return EXIT_SUCCESS;
}

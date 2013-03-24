#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "LogisticRegression.hpp"

using namespace ml;

int main (int argc, char** argv)
{
	LogisticRegression machine;

	// Test case: OR
	Mat2d features = (Mat2d(7, 3) << 
			1, 0, 0, 
			1, 0, 0, 
			1, 0, 0, 
			1, 0, 0, 
			1, 0, 1, 
			1, 1, 0, 
			1, 1, 1);
	Mat2d targets = (Mat2d(7, 1) << 0, 0, 0, 0, 1, 1, 1);

	Mat2d test0 = (Mat2d(1, 3) << 1, 0, 0); // -> 0
	Mat2d test1 = (Mat2d(1, 3) << 1, 0, 1); // -> 1
	Mat2d test2 = (Mat2d(1, 3) << 1, 1, 0); // -> 1
	Mat2d test3 = (Mat2d(1, 3) << 1, 1, 1); // -> 1

	std::cout << "Features:\n" << features << std::endl;
	std::cout << "Targets:\n" << targets << std::endl;

	machine.train(features, targets);

	std::cout << "Params:\n" << machine.get_params() << std::endl;
	// expecting params like [0, 1, 1]

	std::cout << "Predictions:\n" << std::endl;
	std::cout << "00:\t" << machine.predict(test0) << std::endl;
	std::cout << "01:\t" << machine.predict(test1) << std::endl;
	std::cout << "11:\t" << machine.predict(test2) << std::endl;
	std::cout << "10:\t" << machine.predict(test3) << std::endl;

	return EXIT_SUCCESS;
}

#include "Regression.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <complex>
#include <queue>

using namespace std;

// Namespace functions
const double ml::sigmoid (double value)
{
	return 1.0 / (1.0 + exp(-value));
}

// impl header
struct ml::Regression::impl
{
	impl (function<double(double)> hypothesis_function) : hypothesis(hypothesis_function) {};

	// Careful with alpha! A high learning rate will overshoot the optimum and mystify you.
	const double alpha = 0.02;
	const double tolerance = 1e-6;
	const size_t step_limit = 1e4;

	const function<double(double)> hypothesis;

	Matd params;

	void train (const Matd& features, const Matd& targets);
	const double predict (const Matd& features) const;
	const Matd insert_bias (const Matd& features) const;

	// TODO give client choice of descent method
	void batch_descent (const Matd& features, const Matd& targets);
	void stochastic_descent (const Matd& features, const Matd& targets);
};

// impl definition

void ml::Regression::impl::batch_descent (const Matd& features, const Matd& targets)
{
	for (size_t step = 0; step < step_limit; step++)
	{
		// Apply hypothesis
		Matd prod = features * params.t();
		Matd hyp(prod.size());
		transform(begin(prod), end(prod), begin(hyp), hypothesis);

		params += (features.t() * (targets - hyp)).t() * alpha / targets.rows;
		// TODO abort when change in params or cost function is < tolerance
	}
}

void ml::Regression::impl::stochastic_descent (const Matd& features, const Matd& targets)
{
	for (size_t step = 0; step < step_limit; step++)
	{
		// TODO implement shuffle
		for (size_t i = 0; i < targets.rows; i++)
		{
			Matd temp(params.size());
			for (size_t j = 0; j < params.cols; j++)
			{
				double update = features.at<double>(i,j) * (targets.at<double>(i) - hypothesis(params.dot(features.rowRange(i, i+1)))) * alpha / targets.rows;
				temp.at<double>(j) = update;
			}
			params += temp;
		}
	}
	// TODO abort when change in params or cost function is < tolerance
}

const ml::Matd ml::Regression::impl::insert_bias (const Matd& features) const
{
	// Create copy with an extra column of ones as bias
	Matd biased(Matd::ones(features.rows, features.cols + 1));
	features.copyTo(biased.colRange(1, biased.cols));

	// NRVO supported
	return biased;
}

void ml::Regression::impl::train (const Matd& features, const Matd& targets)
{
	assert(targets.cols == 1);
	assert(features.rows == targets.rows);

	Matd biased(insert_bias(features));

	params = Matd::zeros(1, biased.cols);

//	cout << "Stochastic descent" << endl;
//	stochastic_descent(biased, targets);

	cout << "Batch descent" << endl;
	batch_descent(biased, targets);
}

const double ml::Regression::impl::predict (const Matd& features) const
{
	// Assume bias element present
	return hypothesis(params.dot(features));
}

//  Regression definition
ml::Regression::Regression (function<double(double)> hypothesis_function) : pimpl(new Regression::impl(hypothesis_function)) {}
ml::Regression::~Regression () {}

void ml::Regression::train (const Matd& features, const Matd& targets)
{
	return pimpl->train(features, targets);
}

const double ml::Regression::predict (const Matd& features) const
{
	// Insert bias element before prediction
	return pimpl->predict(pimpl->insert_bias(features));
}

const ml::Matd& ml::Regression::get_params () const
{
	return pimpl->params;
}

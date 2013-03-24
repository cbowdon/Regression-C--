#include "LogisticRegression.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <complex>

using namespace std;

// impl header
struct ml::LogisticRegression::impl
{
	// Careful with alpha! A high learning rate will overshoot the optimum and mystify you.
	const double alpha = 0.02;
	const double tolerance = 1e-6;
	const size_t step_limit = 1e4;

	Mat2d params;

	void train (const Mat2d& features, const Mat2d& targets);
	const double predict (const Mat2d& features) const;
	const Mat2d insert_bias (const Mat2d& features) const;

	void batch_descent (const Mat2d& features, const Mat2d& targets);
	void stochastic_descent (const Mat2d& features, const Mat2d& targets);

	const double hypothesis (const Mat2d& params, const Mat2d& features) const;
	const double sigmoid (double value) const;
};

// impl definition

const double ml::LogisticRegression::impl::hypothesis (const Mat2d& params, const Mat2d& features) const
{
	// to do linear regression, just let hypothesis = params . features
	return sigmoid(params.dot(features));
}

const double ml::LogisticRegression::impl::sigmoid (double value) const
{
	return 1.0 / (1.0 + exp(-value));
}

void ml::LogisticRegression::impl::batch_descent (const Mat2d& features, const Mat2d& targets)
{
	for (size_t step = 0; step < step_limit; step++)
	{
		params += (features.t() * (targets - features * params.t())).t() * alpha / targets.rows;
		// TODO adapt this for alternative hypotheses
		// TODO abort when change in params or cost function is < tolerance
	}
}

void ml::LogisticRegression::impl::stochastic_descent (const Mat2d& features, const Mat2d& targets)
{
	for (size_t step = 0; step < step_limit; step++)
	{
		for (size_t i = 0; i < targets.rows; i++)
		{
			Mat2d temp(params.size());
			for (size_t j = 0; j < params.cols; j++)
			{
				double update = features.at<double>(i,j) * (targets.at<double>(i) - hypothesis(params, features.rowRange(i, i+1))) * alpha / targets.rows;
				temp.at<double>(j) = update;
			}
			params += temp;
		}
		// TODO abort when change in params or cost function is < tolerance
	}
}

const ml::Mat2d ml::LogisticRegression::impl::insert_bias (const Mat2d& features) const
{
	// Create copy with an extra column of ones as bias
	Mat2d biased(Mat2d::ones(features.rows, features.cols + 1));
	features.copyTo(biased.colRange(1, biased.cols));

	// NRVO supported
	return biased;
}

void ml::LogisticRegression::impl::train (const Mat2d& features, const Mat2d& targets)
{
	assert(targets.cols == 1);
	assert(features.rows == targets.rows);

	Mat2d biased(insert_bias(features));

	params = Mat2d::zeros(1, biased.cols);

	cout << "Stochastic descent" << endl;
	stochastic_descent(biased, targets);

//	cout << "Batch descent" << endl;
//	batch_descent(biased, targets);
}

const double ml::LogisticRegression::impl::predict (const Mat2d& features) const
{
	// Assume bias element present
	return hypothesis(params, features);
}

// Logistic Regression definition

ml::LogisticRegression::LogisticRegression () : pimpl(new LogisticRegression::impl) {}
ml::LogisticRegression::~LogisticRegression () {}

void ml::LogisticRegression::train (const Mat2d& features, const Mat2d& targets)
{
	return pimpl->train(features, targets);
}

const double ml::LogisticRegression::predict (const Mat2d& features) const
{
	// Insert bias element before prediction
	return pimpl->predict(pimpl->insert_bias(features));
}

const ml::Mat2d& ml::LogisticRegression::get_params () const
{
	return pimpl->params;
}

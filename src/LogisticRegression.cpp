#include "LogisticRegression.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>

using namespace std;

// impl header
struct ml::LogisticRegression::impl
{
	const double alpha = 0.1; // learning rate
	const double tolerance = 1e-5;
	const size_t step_limit = 1e5;

	Mat2d params;

	void train (const Mat2d& features, const Mat2d& targets);
	const double predict (const Mat2d& features) const;

	void batch_descent (const Mat2d& features, const Mat2d& targets);
	void stochastic_descent (const Mat2d& features, const Mat2d& targets);

	const double sigmoid (double value) const; // hypothesis
	const double hyp (double x, double theta) const;
};

// impl definition

void ml::LogisticRegression::impl::batch_descent (const Mat2d& features, const Mat2d& targets)
{
	for (size_t j = 0; j < params.cols; j++)
	{
		double update = 0;
		size_t steps = 0;
		do
		{
			Mat2d guess(targets.size());
			for (size_t m = 0; m < targets.rows; m++)
			{
				guess.at<double>(m) = (targets.at<double>(m) - predict(features.rowRange(m, m + 1))) * features.at<double>(m, j);
			}
			update = alpha * accumulate(begin(guess), end(guess), 0.0);

			params.at<double>(j) += update;
		}
		while (steps++ < step_limit && update > tolerance);
		cout << "Steps: " << steps << ", update: " << update << endl;
	}
	cout << endl;
}

void ml::LogisticRegression::impl::stochastic_descent (const Mat2d& features, const Mat2d& targets)
{
	for (size_t m = 0; m < targets.rows; m++)
	{
		for (size_t i = 0; i < params.cols; i++)
		{
			double update = 0;
			size_t steps = 0;
			do
			{
				update = alpha * (targets.at<double>(m)) - predict(features.rowRange(m, m+1)) * features.at<double>(m, i);
				params.at<double>(i) += update;
			}
			while (steps++ < step_limit && update > tolerance);
			cout << "Steps: " << steps << ", update: " << update << endl;
		}
	}
	cout << endl;
}

const double ml::LogisticRegression::impl::sigmoid (double value) const
{
	return 1.0 / (1.0 + exp(-value));
}

void ml::LogisticRegression::impl::train (const Mat2d& features, const Mat2d& targets)
{
	assert(targets.cols == 1);
	assert(features.rows == targets.rows);

	params = Mat2d::zeros(1, features.cols);
	batch_descent(features, targets);

	params = Mat2d::zeros(1, features.cols);
	stochastic_descent(features, targets);
}

const double ml::LogisticRegression::impl::predict (const Mat2d& features) const
{
	return sigmoid(params.dot(features));
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
	return pimpl->predict(features);
}

const ml::Mat2d& ml::LogisticRegression::get_params () const
{
	return pimpl->params;
}

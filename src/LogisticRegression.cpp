#include "LogisticRegression.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>

using namespace std;

// impl
struct ml::LogisticRegression::impl
{
	const double alpha = 0.1; // learning rate

	Mat2d params;

	void train (const Mat2d features, const Mat2d targets);
	const double predict (const Mat2d features) const;

	const double sigmoid (double value) const; // hypothesis
};

const double ml::LogisticRegression::impl::sigmoid (double value) const
{
	return 1.0 / (1.0 + exp(-value));
}

void ml::LogisticRegression::impl::train (const Mat2d features, const Mat2d targets)
{
	assert(targets.cols == 1);
	assert(features.rows == targets.rows);
}

const double ml::LogisticRegression::impl::predict (const Mat2d features) const
{
	return 0;
}

// Logistic Regression Interface

ml::LogisticRegression::LogisticRegression () : pimpl(new LogisticRegression::impl) {}
ml::LogisticRegression::~LogisticRegression () {}

void ml::LogisticRegression::train (const Mat2d features, const Mat2d targets)
{
	return pimpl->train(features, targets);
}

const double ml::LogisticRegression::predict (const Mat2d features) const
{
	return pimpl->predict(features);
}

#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <memory>
#include <opencv2/core/core.hpp>

namespace ml
{
	typedef cv::Mat_<double> Mat2d;

	// Logistic regression: a simple classifier
	// hypothesis is sigmoid function
	// use gradient descent to figure out params
	class LogisticRegression
	{
		public:
			LogisticRegression ();
			~LogisticRegression ();

			void train (const Mat2d& features, const Mat2d& targets);
			const double predict (const Mat2d& features) const;
			const Mat2d& get_params () const;

		private:
			struct impl;
			std::unique_ptr<impl> pimpl;
	};
};

#endif

#ifndef REGRESSION_H
#define REGRESSION_H

#include <memory>
#include <opencv2/core/core.hpp>
#include <functional>

namespace ml
{
	typedef cv::Mat_<double> Matd;

	//  Regression: using gradient descent to figure out params
	class Regression
	{
		public:
			Regression (std::function<double(double)> hypothesis_function);
			~Regression ();

			void train (const Matd& features, const Matd& targets);
			const double predict (const Matd& features) const;
			const Matd& get_params () const;

		private:
			struct impl;
			std::unique_ptr<impl> pimpl;
	};

	extern const double sigmoid (double value);

	class LogisticRegression : public Regression
	{
		public:
			LogisticRegression () : Regression (sigmoid) {};
	};

	class LinearRegression : public Regression
	{
		public:
			LinearRegression () : Regression ([](double d) { return d; }) {};
	};
};

#endif

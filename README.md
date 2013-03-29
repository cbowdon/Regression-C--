Readme for Regression (C++11)
=============================

After watching a couple of Andrew Ng's lectures on machine learning I was inspired to try write my own implementation of the simplest gradient descent algorithms in (fairly) idiomatic C++11.

Both batch gradient descent and stochastic gradient descent are implemented, and with the appropriate hypothesis functions it's possible to perform linear regression and logistic regression. The simple tests in main.cpp show it working fairly neatly on simple examples like finding the gradient of a line and learning the OR function.

I used OpenCV's linear algebra utilities rather than a dedicated library like Armadillo just because I'm familiar with OpenCV and have it already installed on my machine.

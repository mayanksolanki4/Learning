import numpy as np


def sigmoid(theta_x):
	return 1/(1 + np.exp(-theta_x))

def log_likelihood(X, theta, y):
	ll = 0
	for x, l in zip(X, y):
		theta_x = np.dot(x, theta)
		ll += l*np.log(sigmoid(theta_x)) + (1-l)*np.log(1-sigmoid(theta_x))
	return ll

def stochastic_gradient_descent(X, theta, y, alpha, num_iters):
	for _ in range(num_iters):
		ind = np.random.randint(0, X.shape[0])
		prediction = sigmoid(np.dot(X[ind], theta))
		err = y[ind] - prediction
		theta = theta + alpha*err*X[ind]
	return theta


X = np.loadtxt('data.txt')
y = np.loadtxt('label.txt')
theta = np.random.randn(X.shape[1])

print(log_likelihood(X, theta, y))

theta = stochastic_gradient_descent(X, theta, y, 0.1, 1000)
print(theta)

print(log_likelihood(X, theta, y))



for _ in range(10):
	ind = np.random.randint(0, X.shape[0])
	x = X[ind]
	print(x)
	pred = sigmoid(np.dot(x, theta))
	print(pred)
# Importing the necessary libraries
# add a test here
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Defining the parameters of the Gaussian Mixture Model
mu1 = np.array([1, 0])
cov1 = np.array([[1, 0.5], [0.5, 1]])

mu2 = np.array([6, 6])
cov2 = np.array([[1, 0.5], [0.5, 1]])

# Defining the prior probabilities
pi1 = 0.4
pi2 = 0.6

# Generating the data points
data = np.concatenate((np.random.multivariate_normal(mu1, cov1, int(1000 * pi1)),
                       np.random.multivariate_normal(mu2, cov2, int(1000 * pi2))))

# Plotting the data points
plt.scatter(data[:, 0], data[:, 1])
plt.show()

# Defining the parameters of the Gaussian Mixture Model
n_clusters = 2
n_samples = len(data)

# Initializing the mean, covariance and prior probabilities
mu = np.random.randint(min(data[:, 0]), max(data[:, 0]), size=(n_clusters, 2))
cov = np.zeros((n_clusters, 2, 2))
for i in range(n_clusters):
    cov[i] = np.diag(np.random.randint(1, 8, size=2))
pi = np.ones(n_clusters) / n_clusters


# Defining the probability function
def prob(data, mu, cov, pi):
    n_clusters = len(pi)
    prob = np.zeros((n_clusters, len(data)))
    # For each of the two clusters, we are calculating the probability values as explained in slide 44
    # equation and slide 45 equation 1. This is in two dimension equivalent of 1 dimension e.g. shown in slide 43
    # first equation.
    # We are using scipy.stats multivariate_normal.pdf that takes the data(in 2D), mean and covariance
    # of the given gaussian density distribution to calculate the probability.
    for i in range(n_clusters):
        prob[i] = pi[i] * multivariate_normal.pdf(data, mu[i], cov[i])
    return prob


# Defining the Expectation-Maximization Algorithm
def EM(data, mu, cov, pi):
    n_clusters = len(pi)
    n_samples = len(data)
    log_likelihoods = []

    # Iterating till convergence
    for i in range(1000):
        # Calculating the probabilities
        # The details are explained in the comments for the function above.
        probabilities = prob(data, mu, cov, pi)

        # Calculating the log-likelihood
        log_likelihood = np.sum(np.log(np.sum(probabilities, axis=0)))

        log_likelihoods.append(log_likelihood)

        # Calculating the Expectation Step
        # This is implementation of equation 2 in slide 43 (1D e.g) and equation 2 in slide 45 (since this is 2d)
        weights = probabilities / np.sum(probabilities, axis=0)

        # Calculating the Maximization Step
        # This is implementation of line 4 and line 6 from slide 43. We are calculating the adjusted mean based on
        # the probabilities calculated from previous step.
        mu = np.dot(weights, data) / np.sum(weights, axis=1).reshape(-1, 1)

        # We are initializing the cov 2 d array with zeroes.

        cov = np.zeros((n_clusters, 2, 2))
        # This is implementation of line 5 and line 7 from slide 43. We are calculating the adjusted covariance
        # using the probabilities and means calculated from previous steps and sample data values.
        for j in range(n_clusters):
            x = data - mu[j]
            cov[j] = np.dot(weights[j] * x.T, x) / np.sum(weights[j])

        # This is calculation for the prior probabilities calculation  is to be used for calculation of probabilities
        # later in the flow.
        pi = np.sum(weights, axis=1) / n_samples

    return mu, cov, pi, log_likelihoods


# Running the Expectation-Maximization Algorithm
mu, cov, pi, log_likelihoods = EM(data, mu, cov, pi)

# Plotting the log-likelihoods
plt.plot(log_likelihoods)
plt.show()

# Plotting the clusters
plt.scatter(data[:, 0], data[:, 1])
for i in range(n_clusters):
    x, y = np.random.multivariate_normal(mu[i], cov[i], 100).T
    plt.scatter(x, y)
plt.show()

import numpy

from sklearn.neighbors import NearestNeighbors

import getpass

username = getpass.getuser()

import sys

sys.path.append('/Users/{}/Documents/Reference/G/github/sidpy/sidpy'.format(username))

import sidpy

def simulate_ls(N_sim, x, p_opt, k = None):
	if k is None:
		k = int(numpy.power(x.shape[0] - p_opt, 4./(4 + 1 + p_opt)))

		print("Using k = {}.".format(k))

	N = x.shape[0]

	X = sidpy.embed_ts(x, p_max = p_opt)

	x_boot = numpy.zeros(N_sim)

	J = numpy.random.randint(p_opt, N)

	x_boot[:p_opt] = x[J-p_opt:J]

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X[:, :-1])

	for t in range(N_sim - p_opt):
		x_cur = x_boot[t:t + p_opt]

		distances, indices = nbrs.kneighbors(x_cur.reshape(1, -1))

		x_boot[t + p_opt] = X[numpy.random.choice(indices[0][1:], size = 1), -1]
		# x_boot[t + p_opt] = numpy.random.randn(1)*0.025 + X[numpy.random.choice(indices[0][1:], size = 1), -1]

	return x_boot

def simulate_rap(N_sim, x, p_opt, k = None):
	if k is None:
		k = int(numpy.power(x.shape[0] - p_opt, 4./(4 + 1 + p_opt)))

		print("Using k = {}.".format(k))

	N = x.shape[0]

	X = sidpy.embed_ts(x, p_max = p_opt)

	x_boot = numpy.zeros(N_sim)

	J = numpy.random.randint(p_opt, N)

	x_boot[:p_opt] = x[J-p_opt:J]

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X[:, :-1])

	for t in range(N_sim - p_opt):
		x_cur = x_boot[t:t + p_opt]

		distances, indices = nbrs.kneighbors(x_cur.reshape(1, -1))

		distances = distances.flatten()
		indices = indices.flatten()

		sample_prob = 1/distances # Sample weighted by the reciprocal of the distance to the target point.

		sample_prob = sample_prob/numpy.sum(sample_prob) # Normalize the probability.

		which_nn = numpy.random.choice(indices, size = 1, p = sample_prob)

		delta_resid = X[which_nn, -1] - X[which_nn, -2]

		x_boot[t + p_opt] = x_boot[t + p_opt - 1] + delta_resid

	return x_boot
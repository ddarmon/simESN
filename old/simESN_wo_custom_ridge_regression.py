import numpy
import scipy
import ipdb
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

import getpass

username = getpass.getuser()

import sys

sys.path.append('/Users/{}/Documents/Reference/G/github/sidpy/sidpy'.format(username))

import sidpy

def learn_esn(x, N_res = 400, rho = 0.99, alpha = 0.1):
	p_max = 1

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set up data structures for the echo state network.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	X = numpy.matrix(sidpy.embed_ts(x, p_max = p_max).T)

	# N_res = 100 # 'Works' with slogistic and stent when p_max = 1.
	# N_res = 150
	# N_res = 200 # Works with slorenz
	# N_res = 300
	# N_res = 400
	# N_res = 800
	# N_res = 1000

	Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

	Win  = 2*(numpy.random.rand(N_res, p_max) - 0.5)
	W    = 2*(numpy.random.rand(N_res, N_res) - 0.5)
	Wfb  = 2*(numpy.random.rand(N_res, 1) - 0.5)
	bias_constant = 2*(numpy.random.rand(1) - 0.5)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Generate W_skeleton as an Erdos-Renyi graph with some
	# fixed degree.
	# 
	# Recall that the expected mean degree of an Erdos 
	# renyi graph is d = n*p, so we should take 
	# p = d/n
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# W_skeleton = numpy.random.rand(N_res, N_res)

	# # mean_degree = 3
	# mean_degree = 10
	# p_erdosrenyi = mean_degree/float(N_res)

	# W_skeleton[W_skeleton <= p_erdosrenyi] = 1
	# W_skeleton[W_skeleton != 1] = 0

	# W = W*W_skeleton

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	s = numpy.max(numpy.abs(numpy.linalg.eigvals(W)))

	###
	# rho = spectral radius of W
	###
	# rho = 0.50
	# rho = 0.80
	rho = 0.99
	# rho = 1.20
	# rho = 1.45
	W = rho*W/s

	print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		Y[:, t] = numpy.tanh(numpy.dot(Win, X[:-1, t]) + numpy.dot(W, Y[:, t-1]) + bias_constant)

	print("Done running ESN with time series as input:")

	print("Estimating output weights:")

	# Using Ridge Regression:

	vec_ones = numpy.ones(X.shape[1]).reshape(1, -1)

	S = numpy.row_stack((vec_ones, X[:-1, :], Y)).T

	# alpha = 0.01 # Weak regularization.
	alpha = 0.1 # Works well for most systems
	# alpha = 1.0 # The regularization parameter, default value
	# alpha = 10.
	# alpha = 1000. # Strong regularization

	ridge_reg = Ridge(alpha=alpha,fit_intercept=True)
	ridge_reg.fit(S[:, 1:], numpy.ravel(X[-1, :]))

	Wout = numpy.array([ridge_reg.intercept_] + ridge_reg.coef_.tolist())

	Xhat = numpy.dot(S, Wout)

	print("Done estimating output weights:")

	#
	####################################################


	x_esn = numpy.ravel(Xhat)

	err_esn = x[p_max:] - x_esn

	return x_esn, X, Y, err_esn, Win, W, Wout, bias_constant

# This version allows for p_max >= 1, which is *not* the general definition
# of the ESN.
def simulate_from_esn_old(N_sim, X, Y, err_esn, Win, W, Wout, bias_constant, p_max = 1, print_iter = False):
	N_res = Y.shape[0]

	J = numpy.random.randint(1, X.shape[1])

	x_esn_sim = numpy.zeros(N_sim)
	x_esn_sim[:p_max] = X[:-1, J].T

	Y_sim = numpy.matrix(numpy.zeros((N_res, 1)))

	#### DOUBLE CHECK THIS ####
	Y_sim = Y[:, J-1]

	####################################################
	# I am not sure about the indexing here:

	for t in range(p_max, N_sim):
		if t % 10000 == 0:
			if print_iter:
				print("On iterate {} of {}.".format(t + 1, N_sim))

		Y_sim = numpy.tanh(numpy.dot(Win, x_esn_sim[t-p_max:t].reshape(-1, 1)) + numpy.dot(W, Y_sim) + bias_constant)

		x_esn_sim[t] = float(numpy.dot(numpy.row_stack(([1], x_esn_sim[t-p_max:t].reshape(-1, 1), Y_sim)).T, Wout))
		x_esn_sim[t] += numpy.random.choice(err_esn) # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

	#
	####################################################

	plt.figure()
	plt.plot(x_esn_sim[:1000])
	plt.plot(X[1, :].T)
	plt.show()

	plt.figure()
	plt.scatter(x_esn_sim[:-1], x_esn_sim[1:])

	ipdb.set_trace()

	return x_esn_sim

def simulate_from_esn(N_sim, X, Y, err_esn, Win, W, Wout, bias_constant, p_max = 1, print_iter = False):
	N_res = Y.shape[0]

	Wout = Wout.reshape(-1, 1) # (2 + Nres) x 1
	W = W.T
	Win = Win.T

	J = numpy.random.randint(1, X.shape[1])

	x_esn_sim = numpy.zeros(N_sim)
	x_esn_sim[0] = X[0, J].T

	#### DOUBLE CHECK THIS ####
	Y_sim = Y[:, J-1].T # 1 x Nres

	####################################################
	# I am not sure about the indexing here:

	vec_for_mult = numpy.column_stack(([1], x_esn_sim[0].reshape(-1, 1), Y_sim)) # 1 x (2 + Nres)

	# ipdb.set_trace()

	for t in range(p_max, N_sim):
		if t % 10000 == 0:
			if print_iter:
				print("On iterate {} of {}.".format(t + 1, N_sim))


		Y_sim = numpy.tanh(Win*x_esn_sim[t-1] + numpy.dot(Y_sim, W) + bias_constant)

		vec_for_mult[0, 1] = x_esn_sim[t-1]
		vec_for_mult[0, 2:] = Y_sim[0, :]

		x_esn_sim[t] = float(numpy.dot(vec_for_mult, Wout))
		x_esn_sim[t] += numpy.random.choice(err_esn) # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

		# ipdb.set_trace()

	#
	####################################################

	# plt.figure()
	# plt.plot(x_esn_sim[:1000])
	# plt.plot(X[1, :].T)

	# plt.figure()
	# plt.scatter(x_esn_sim[:-1], x_esn_sim[1:])

	# plt.show()

	# ipdb.set_trace()

	return x_esn_sim

def learn_esn_umd(x, p_max = 1, N_res = 400, rho = 0.99, alpha = 0.1):
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set up data structures for the echo state network.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	X = numpy.matrix(sidpy.embed_ts(x, p_max = p_max).T)

	# N_res = 100 # 'Works' with slogistic and stent when p_max = 1.
	# N_res = 150
	# N_res = 200 # Works with slorenz
	# N_res = 300
	# N_res = 400
	# N_res = 800
	# N_res = 1000

	Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

	Win  = 2*(numpy.random.rand(N_res, p_max) - 0.5)
	W    = 2*(numpy.random.rand(N_res, N_res) - 0.5)
	Wfb  = 2*(numpy.random.rand(N_res, 1) - 0.5)
	bias_constant = 2*(numpy.random.rand(1) - 0.5)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Generate W_skeleton as an Erdos-Renyi graph with some
	# fixed degree.
	# 
	# Recall that the expected mean degree of an Erdos 
	# renyi graph is d = n*p, so we should take 
	# p = d/n
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# W_skeleton = numpy.random.rand(N_res, N_res)

	# # mean_degree = 3
	# mean_degree = 10
	# p_erdosrenyi = mean_degree/float(N_res)

	# W_skeleton[W_skeleton <= p_erdosrenyi] = 1
	# W_skeleton[W_skeleton != 1] = 0

	# W = W*W_skeleton

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	s = numpy.max(numpy.abs(numpy.linalg.eigvals(W)))

	###
	# rho = spectral radius of W
	###
	# rho = 0.50
	# rho = 0.80
	rho = 0.99
	# rho = 1.20
	# rho = 1.45
	W = rho*W/s

	print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		Y[:, t] = numpy.tanh(numpy.dot(Win, X[:-1, t]) + numpy.dot(W, Y[:, t-1]) + bias_constant)

	print("Done running ESN with time series as input:")

	print("Estimating output weights:")

	# Using Ridge Regression:

	vec_ones = numpy.ones(X.shape[1]).reshape(1, -1)

	S = numpy.row_stack((vec_ones, Y)).T

	# alpha = 0.01 # Weak regularization.
	alpha = 0.1 # Works well for most systems
	# alpha = 1.0 # The regularization parameter, default value
	# alpha = 10.
	# alpha = 1000. # Strong regularization

	ridge_reg = Ridge(alpha=alpha,fit_intercept=True)
	ridge_reg.fit(S[:, 1:], numpy.ravel(X[-1, :]))

	Wout = numpy.array([ridge_reg.intercept_] + ridge_reg.coef_.tolist())

	Xhat = numpy.dot(S, Wout)

	print("Done estimating output weights:")

	#
	####################################################


	x_esn = numpy.ravel(Xhat)

	err_esn = x[p_max:] - x_esn

	return x_esn, X, Y, err_esn, Win, W, Wout, bias_constant

def simulate_from_esn_umd(N_sim, X, Y, err_esn, Win, W, Wout, bias_constant, p_max = 1, is_stochastic = True, print_iter = False):
	N_res = Y.shape[0]

	Wout = Wout.reshape(-1, 1) # (2 + Nres) x 1

	J = numpy.random.randint(1, X.shape[1])

	x_esn_sim = numpy.zeros(N_sim)
	# x_esn_sim[:p_max] = X[J, :-1]
	x_esn_sim[:p_max] = numpy.ravel(X[:-1, J])

	#### DOUBLE CHECK THIS ####
	Y_sim = Y[:, J] # 1 x Nres

	####################################################
	# I am not sure about the indexing here:

	vec_for_mult = numpy.column_stack(([1], Y_sim.T)) # 1 x (2 + Nres)

	# ipdb.set_trace()

	for t in range(p_max, N_sim):
		if t % 10000 == 0:
			if print_iter:
				print("On iterate {} of {}.".format(t + 1, N_sim))

		Y_sim = numpy.tanh(numpy.dot(Win, x_esn_sim[t-p_max:t].reshape(-1, 1)) + numpy.dot(W, Y_sim) + bias_constant)

		vec_for_mult[0, 1:] = Y_sim[:, 0].T

		x_esn_sim[t] = float(numpy.dot(vec_for_mult, Wout))
		if is_stochastic:
			x_esn_sim[t] += numpy.random.choice(err_esn) # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

	# ipdb.set_trace()

	#
	####################################################

	# plt.figure()
	# plt.plot(x_esn_sim[:1000])
	# plt.plot(X[1, :].T)

	# plt.figure()
	# plt.scatter(x_esn_sim[:-1], x_esn_sim[1:])

	# plt.show()

	# ipdb.set_trace()

	return x_esn_sim

def learn_esn_umd_sparse(x, p_max = 1, N_res = 400, rho = 0.99, alpha = 0.1, Win_scale = 1.):
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set up data structures for the echo state network.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	X = numpy.matrix(sidpy.embed_ts(x, p_max = p_max).T)

	# N_res = 100 # 'Works' with slogistic and stent when p_max = 1.
	# N_res = 150
	# N_res = 200 # Works with slorenz
	# N_res = 300
	# N_res = 400
	# N_res = 800
	# N_res = 1000

	Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

	Win  = Win_scale*2*(numpy.random.rand(N_res, p_max) - 0.5)
	bias_constant = 2*(numpy.random.rand(1) - 0.5)

	# mean_degree = 3
	mean_degree = 10
	p_erdosrenyi = mean_degree/float(N_res)

	W = scipy.sparse.random(m = N_res, n = N_res, density = p_erdosrenyi, data_rvs = scipy.stats.uniform(loc = -0.5, scale = 1).rvs)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	###
	# rho = spectral radius of W
	###
	# rho = 0.50
	# rho = 0.80
	rho = 0.99
	# rho = 1.20
	# rho = 1.45

	s = scipy.sparse.linalg.svds(W, k=1)

	W = W.multiply(numpy.abs(rho/float(s[1])))

	print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		Y[:, t] = numpy.tanh(numpy.dot(Win, X[:-1, t]) + W.dot(Y[:, t-1]) + bias_constant)

	print("Done running ESN with time series as input:")

	print("Estimating output weights:")

	# Using Ridge Regression:

	vec_ones = numpy.ones(X.shape[1]).reshape(1, -1)

	S = numpy.row_stack((vec_ones, Y)).T

	# alpha = 0.01 # Weak regularization.
	alpha = 0.1 # Works well for most systems
	# alpha = 1.0 # The regularization parameter, default value
	# alpha = 10.
	# alpha = 1000. # Strong regularization

	ridge_reg = Ridge(alpha=alpha,fit_intercept=True)
	ridge_reg.fit(S[:, 1:], numpy.ravel(X[-1, :]))

	Wout = numpy.array([ridge_reg.intercept_] + ridge_reg.coef_.tolist())

	Xhat = numpy.dot(S, Wout)

	print("Done estimating output weights:")

	#
	####################################################


	x_esn = numpy.ravel(Xhat)

	err_esn = x[p_max:] - x_esn

	return x_esn, X, Y, err_esn, Win, W, Wout, bias_constant

def simulate_from_esn_umd_sparse(N_sim, X, Y, err_esn, Win, W, Wout, bias_constant, p_max = 1, is_stochastic = True, print_iter = False):
	N_res = Y.shape[0]

	Wout = Wout.reshape(-1, 1) # (2 + Nres) x 1

	J = numpy.random.randint(1, X.shape[1])

	x_esn_sim = numpy.zeros(N_sim)
	# x_esn_sim[:p_max] = X[J, :-1]
	x_esn_sim[:p_max] = numpy.ravel(X[:-1, J])

	#### DOUBLE CHECK THIS ####
	Y_sim = Y[:, J] # 1 x Nres

	####################################################
	# I am not sure about the indexing here:

	vec_for_mult = numpy.column_stack(([1], Y_sim.T)) # 1 x (2 + Nres)

	# ipdb.set_trace()

	for t in range(p_max, N_sim):
		if t % 10000 == 0:
			if print_iter:
				print("On iterate {} of {}.".format(t + 1, N_sim))

		Y_sim = numpy.tanh(numpy.dot(Win, x_esn_sim[t-p_max:t].reshape(-1, 1)) + W.dot(Y_sim) + bias_constant)

		vec_for_mult[0, 1:] = Y_sim[:, 0].T

		x_esn_sim[t] = float(numpy.dot(vec_for_mult, Wout))
		if is_stochastic:
			x_esn_sim[t] += numpy.random.choice(err_esn) # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

	# ipdb.set_trace()

	#
	####################################################

	return x_esn_sim

def learn_io_esn_umd(y, x, qp_opt = (1, 1), N_res = 400, rho = 0.99, alpha = 0.1):
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set up data structures for the echo state network.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	q_opt = qp_opt[0]
	p_opt = qp_opt[1]

	qp_max = numpy.max(qp_opt)
	qp_sum = numpy.sum(qp_opt)

	# X and Y are (px + 1, py + 1) x (T - m, T - m)
	# where m = max([px, py])

	Y = numpy.matrix(sidpy.embed_ts(y, p_max = qp_max).T)
	X = numpy.matrix(sidpy.embed_ts(x, p_max = qp_max).T)

	Y = Y[qp_max-q_opt:, :]
	X = X[qp_max-p_opt:, :]

	U = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

	Win  = 2*(numpy.random.rand(N_res, qp_sum) - 0.5)
	W    = 2*(numpy.random.rand(N_res, N_res) - 0.5)
	bias_constant = 2*(numpy.random.rand(1) - 0.5)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Generate W_skeleton as an Erdos-Renyi graph with some
	# fixed degree.
	# 
	# Recall that the expected mean degree of an Erdos 
	# renyi graph is d = n*p, so we should take 
	# p = d/n
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# W_skeleton = numpy.random.rand(N_res, N_res)

	# # mean_degree = 3
	# mean_degree = 10
	# p_erdosrenyi = mean_degree/float(N_res)

	# W_skeleton[W_skeleton <= p_erdosrenyi] = 1
	# W_skeleton[W_skeleton != 1] = 0

	# W = W*W_skeleton

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	s = numpy.max(numpy.abs(numpy.linalg.eigvals(W)))

	###
	# rho = spectral radius of W
	###

	W = rho*W/s

	print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		U[:, t] = numpy.tanh(numpy.dot(Win, numpy.row_stack((Y[:-1, t], X[:-1, t]))) + numpy.dot(W, U[:, t-1]) + bias_constant)

	# plt.imshow(U.T, aspect = 0.1)

	print("Done running ESN with time series as input:")

	print("Estimating output weights:")

	# Using Ridge Regression:

	vec_ones = numpy.ones(X.shape[1]).reshape(1, -1)

	S = numpy.row_stack((vec_ones, U)).T

	target = numpy.row_stack((Y[-1, :], X[-1, :])).T

	ridge_reg = Ridge(alpha=alpha,fit_intercept=True)
	ridge_reg.fit(S[:, 1:], target)

	Wout = numpy.column_stack((ridge_reg.intercept_, ridge_reg.coef_)).T

	Zhat = numpy.dot(S, Wout)

	print("Done estimating output weights:")

	#
	####################################################


	y_esn = numpy.ravel(Zhat[:, 0])
	x_esn = numpy.ravel(Zhat[:, 1])

	err_esn_y = y[qp_max:] - y_esn
	err_esn_x = x[qp_max:] - x_esn

	return y_esn, x_esn, Y, X, U, err_esn_y, err_esn_x, Win, W, Wout, bias_constant

def simulate_from_io_esn_umd(N_sim, Y, X, U, err_esn_y, err_esn_x, Win, W, Wout, bias_constant, qp_opt = (1, 1), is_stochastic = True, print_iter = False):
	q_opt = qp_opt[0]
	p_opt = qp_opt[1]

	qp_max = numpy.max(qp_opt)
	qp_sum = numpy.sum(qp_opt)

	N_res = Y.shape[0]

	J = numpy.random.randint(1, X.shape[1])

	z_esn_sim = numpy.zeros((2, N_sim))
	z_esn_sim[0, :q_opt] = numpy.ravel(Y[:-1, J])
	z_esn_sim[1, :p_opt] = numpy.ravel(X[:-1, J])

	#### DOUBLE CHECK THIS ####
	U_sim = U[:, J] # 1 x Nres

	####################################################
	# I am not sure about the indexing here:

	vec_for_mult = numpy.column_stack(([1], U_sim.T)) # 1 x (2 + Nres)

	for t in range(qp_max, N_sim):
		if t % 10000 == 0:
			if print_iter:
				print("On iterate {} of {}.".format(t + 1, N_sim))

		Zt = numpy.concatenate((z_esn_sim[0, t-q_opt:t], z_esn_sim[1, t-p_opt:t])).reshape(-1, 1)

		U_sim = numpy.tanh(numpy.dot(Win, Zt) + numpy.dot(W, U_sim) + bias_constant)

		vec_for_mult[0, 1:] = U_sim[:, 0].T

		z_esn_sim[:, t] = numpy.dot(vec_for_mult, Wout)
		if is_stochastic:
			K = numpy.random.choice(len(err_esn_y))
			z_esn_sim[0, t] += err_esn_y[K] # Add noise sampled from the training set noise if assuming a stochastic dynamical system.
			z_esn_sim[1, t] += err_esn_x[K] # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

	#
	####################################################

	return z_esn_sim[0, :], z_esn_sim[1, :]
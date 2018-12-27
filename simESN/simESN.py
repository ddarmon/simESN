import numpy
import scipy
import matplotlib.pyplot as plt

from scipy.special import expit

from sklearn.linear_model import Ridge
from sklearn import neighbors

import sys

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

def learn_esn_umd_sparse(x, p_max = 1, N_res = 400, rho = 0.99, Win_scale = 1., multi_bias = False, to_plot_regularization = False, output_verbose = False, Win = None, bias_constant = None, W = None, seed_for_ic = None, renormalize_by = 'svd'):

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set up data structures for the echo state network.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	if Win is None:
		Win  = Win_scale*2*(numpy.random.rand(N_res, p_max) - 0.5)
	else:
		assert N_res == Win.shape[0], "Warning: N_res != Win.shape[0]. Change N_res to match Win.shape[0]"

	if bias_constant is None:
		if multi_bias == True:
			bias_constant = 2*(numpy.random.rand(N_res).reshape(-1, 1) - 0.5)
		else:
			bias_constant = 2*(numpy.random.rand(1) - 0.5)
	else:
		if multi_bias == True:
			assert N_res == bias_constant.shape[0], "Warning: N_res != bias_constant.shape[0]. Change N_res to match bias_constant.shape[0]."

	if W is None:
		# mean_degree = 3
		mean_degree = 10
		p_erdosrenyi = mean_degree/float(N_res)

		W = scipy.sparse.random(m = N_res, n = N_res, density = p_erdosrenyi, data_rvs = scipy.stats.uniform(loc = -0.5, scale = 1).rvs)
	else:
		assert N_res == W.shape[0], "Warning: N_res != W.shape[0]. Change N_res to match W.shape[0]."

	X = numpy.matrix(sidpy.embed_ts(x, p_max = p_max).T)

	if seed_for_ic is not None:
		rng_state = numpy.random.get_state()

		numpy.random.seed(seed_for_ic)

		Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

		numpy.random.set_state(rng_state)
	else:
		Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# ipdb.set_trace()

	if renormalize_by == 'svd': # Normalize by the largest singular value of W.
		s = scipy.sparse.linalg.svds(W, k=1)

		W = W.multiply(numpy.abs(rho/float(s[1])))
	elif renormalize_by == 'eigen': # Normalize by the spectral radius of the W.
		lam = scipy.sparse.linalg.eigs(W, k=1)

		W = W.multiply(numpy.abs(rho/float(numpy.abs(lam[0]))))

	if output_verbose:
		print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		Y[:, t] = numpy.tanh(numpy.dot(Win, X[:-1, t]) + W.dot(Y[:, t-1]) + bias_constant)

	if output_verbose:
		print("Done running ESN with time series as input:")

		print("Estimating output weights:")

	Wout, x_esn, err_esn = estimate_ridge_regression_w_splithalf_cv(X.T, Y.T, to_plot = to_plot_regularization)

	return x_esn, X, Y, err_esn, Win, W, Wout, bias_constant

def learn_esn_hybrid_sparse(x, expert_info, p_max = 1, N_res = 400, rho = 0.99, Win_scale = 1., multi_bias = False, to_plot_regularization = False, output_verbose = False, Win = None, bias_constant = None, W = None, seed_for_ic = None):
	"""
	Predict the future of x using the past of x via an echo state network.
	We also include a stream of expert information, in this case the logit
	of the predictive probability provided by an epsilon-machine inferred
	using x.

	NOTE: expert_info is both used to drive the ESN and as an input to the
	output layer of the ESN.

	Parameters
	----------
	x : numpy.array
			The time series to use as the input / output of the ESN.
	expert_info : numpy.array
			The 'expert information' time series also pass to the
			ESN.
	p_max : int
			The model order used to embed x into X.
	N_res : int
			The number of reservoir nodes in the ESN.
	rho : float
			The desired spectral radius of the inter-node
			matrix of the ESN.
	Win_scale : float
			The amount to scale the random projections in the input
			matrix.
	multi_bias : boolean
			Whether (True) or not (False) to include node-specific
			biases.
	to_plot_regularization : boolean
			Whether to plot the regression coefficients as a function
			of the reguarlization parameter of the ridge regression.
	output_verbose : boolean
			Whether to print the stages of learning the ESN.
	Win : numpy.array
			The input matrix to the ESN. If None, randomly generated
			using the parameters given previously.
	bias_constant : numpy.array
			The bias constant of each reservoir node. If None, generated
			uniformly on [-1, 1].
	W : numpy.array
			The inter-node matrix of the ESN. If None, randomly generated
			using the parameters given previously.
	seed_for_ic : int
			The seed used to initialize the pseudo-random number generator
			(PRNG) before creating the initial conditions of the reservoir nodes.
			If None, the state of the PRNG is left as-is.

	Returns
	-------
	x_esn : numpy.array
			The predicted values of x, starting at x[p_max]
	X : numpy.array
			The embedded time series, using a lag of p_max.
	Y : numpy.array
			The states of the reservoir nodes.
	Win : numpy.array
			The input matrix to the ESN.
	W : numpy.array
			The inter-node matrix of the ESN.
	Wout : numpy.array
			The learned output matrix.
			Note: This includes the intercept term as the first
			entry.
	bias_constant : numpy.array
			The bias constant of each reservoir node.


	"""


	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Set up data structures for the echo state network.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# Generate each entry of Win independently and uniformly
	# on [-Win_scale, Win_scale]:

	if Win is None:
		Win  = Win_scale*2*(numpy.random.rand(N_res, p_max+1) - 0.5)
	else:
		assert N_res == Win.shape[0], "Warning: N_res != Win.shape[0]. Change N_res to match Win.shape[0]"


	# Generate each bias term independently and uniformly
	# on [-1, 1]:

	if bias_constant is None:
		if multi_bias == True:
			bias_constant = 2*(numpy.random.rand(N_res).reshape(-1, 1) - 0.5)
		else:
			bias_constant = 2*(numpy.random.rand(1) - 0.5)
	else:
		if multi_bias == True:
			assert N_res == bias_constant.shape[0], "Warning: N_res != bias_constant.shape[0]. Change N_res to match bias_constant.shape[0]."

	# Generate the inter-node weight matrix as an Erdos-Renyi random graph with 
	# average degree mean_degree, and weights uniform on [-1/2, 1/2]

	if W is None:
		# mean_degree = 3
		mean_degree = 10
		p_erdosrenyi = mean_degree/float(N_res)

		W = scipy.sparse.random(m = N_res, n = N_res, density = p_erdosrenyi, data_rvs = scipy.stats.uniform(loc = -0.5, scale = 1).rvs)
	else:
		assert N_res == W.shape[0], "Warning: N_res != W.shape[0]. Change N_res to match W.shape[0]."

	# Embed both x and the expert_info into a (T - p_max, p_max + 1) matrix
	# suitable to use as a data matrix for the regression.

	X = numpy.matrix(sidpy.embed_ts(x, p_max = p_max).T)

	E = numpy.matrix(sidpy.embed_ts(expert_info, p_max = p_max).T)

	# Add the expert info to X to act as a direct input to the ESN.

	X_w_expert = numpy.row_stack((E[-1, :], X))

	X = X_w_expert

	# Allow for fixing the seed for generating the initial
	# conditions of the reservoir nodes.

	if seed_for_ic is not None:
		rng_state = numpy.random.get_state()

		numpy.random.seed(seed_for_ic)

		Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

		numpy.random.set_state(rng_state)
	else:
		Y = numpy.matrix(numpy.random.rand(N_res, X.shape[1]))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	# 
	# NOTE: This is a 
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	s = scipy.sparse.linalg.svds(W, k=1)

	W = W.multiply(numpy.abs(rho/float(s[1])))

	# Run the Echo State Network using X as the input.

	if output_verbose:
		print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		Y[:, t] = numpy.tanh(numpy.dot(Win, X[:-1, t]) + W.dot(Y[:, t-1]) + bias_constant)

	if output_verbose:
		print("Done running ESN with time series as input:")

		print("Estimating output weights:")

	# Harvest the reservoir states and use them for prediction.

	# Remember: we are explicitly stacking on the expert info into the Y matrix, so that we
	# can explicitly learn whether we should just use the expert info, just use the reservoir
	# state info, or use some combination of the two.

	# We will stack on the expert info as *probabilities*, rather than *logit probabilities*,
	# to deal with +/- infinities that can occur when p = 0 or 1.

	Wout, x_esn, err_esn = estimate_ridge_regression_w_splithalf_cv(X.T, numpy.row_stack((Y, expit(E[-1, :]))).T, to_plot = to_plot_regularization)

	# This is what I originally had, where I did *not* explicit include the expert info in the
	# output layer:

	# Wout, x_esn, err_esn = estimate_ridge_regression_w_splithalf_cv(X.T, Y.T, to_plot = to_plot_regularization)

	return x_esn, X, Y, err_esn, Win, W, Wout, bias_constant

def simulate_from_esn_umd_sparse(N_sim, X, Y, err_esn, Win, W, Wout, bias_constant, p_max = 1, is_stochastic = True, knn_errs = False, nn_number = None, print_iter = False):

	if knn_errs == True:
		X_knn = X.T[:, :-1]

		if nn_number == None:
			nn_number = int(numpy.power(X_knn.shape[0], 4./(X_knn.shape[1] + 1 + 4)))

		knn = neighbors.NearestNeighbors(nn_number, algorithm = 'kd_tree', p = 2.)

		knn_out = knn.fit(X_knn)

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
			if knn_errs == True:
				distances, neighbor_inds = knn_out.kneighbors(x_esn_sim[t-p_max:t].reshape(1, -1))

				err_term = err_esn[numpy.random.choice(numpy.ravel(neighbor_inds), size = 1)]
			else:
				err_term = numpy.random.choice(err_esn, size = 1)
			x_esn_sim[t] += err_term # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

	# ipdb.set_trace()

	#
	####################################################

	return x_esn_sim

def learn_io_esn_umd_sparse(y, x, qp_opt = (1, 1), N_res = 400, rho = 0.99, multi_bias = False, to_plot_regularization = False):
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

	if multi_bias:
		bias_constant = 2*(numpy.random.rand(1).reshape(-1, 1) - 0.5)
	else:
		bias_constant = 2*(numpy.random.rand(1) - 0.5)

	mean_degree = 10
	p_erdosrenyi = mean_degree/float(N_res)

	W = scipy.sparse.random(m = N_res, n = N_res, density = p_erdosrenyi, data_rvs = scipy.stats.uniform(loc = -0.5, scale = 1).rvs)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Normalize W so it has condition number of rho:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	s = scipy.sparse.linalg.svds(W, k=1)

	W = W.multiply(numpy.abs(rho/float(s[1])))

	print("Running ESN with time series as input:")

	for t in range(1, X.shape[1]):
		U[:, t] = numpy.tanh(numpy.dot(Win, numpy.row_stack((Y[:-1, t], X[:-1, t]))) + W.dot(U[:, t-1]) + bias_constant)

	# plt.imshow(U.T, aspect = 0.1)

	print("Done running ESN with time series as input:")

	print("Estimating output weights:")

	# Using Ridge Regression:

	target = numpy.row_stack((Y[-1, :], X[-1, :])).T

	Wout, y_esn, x_esn, err_esn_y, err_esn_x = estimate_ridge_regression_joint_w_splithalf_cv(target, U.T, to_plot = to_plot_regularization)

	return y_esn, x_esn, Y, X, U, err_esn_y, err_esn_x, Win, W, Wout, bias_constant

def simulate_from_io_esn_umd_sparse(N_sim, Y, X, U, err_esn_y, err_esn_x, Win, W, Wout, bias_constant, qp_opt = (1, 1), is_stochastic = True, print_iter = False):
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

		U_sim = numpy.tanh(numpy.dot(Win, Zt) + W.dot(U_sim) + bias_constant)

		vec_for_mult[0, 1:] = U_sim[:, 0].T

		z_esn_sim[:, t] = numpy.dot(vec_for_mult, Wout)
		if is_stochastic:
			K = numpy.random.choice(len(err_esn_y))
			z_esn_sim[0, t] += err_esn_y[K] # Add noise sampled from the training set noise if assuming a stochastic dynamical system.
			z_esn_sim[1, t] += err_esn_x[K] # Add noise sampled from the training set noise if assuming a stochastic dynamical system.

	#
	####################################################

	return z_esn_sim[0, :], z_esn_sim[1, :]

def estimate_ridge_regression_w_splithalf_cv(X_ridge, Y_ridge, to_plot = False, is_verbose = False):
	"""
	Estimate the coefficients of a linear model X = f(Y) using split-half cross-validated ridge regression.

	In this case, X_ridge is the state of the dynamical system stacked into a data matrix, and Y is the
	reservoir states.

	Parameters
	----------
	var1 : type
			description

	Returns
	-------
	var1 : type
			description

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	N_res = Y_ridge.shape[1]

	N_train = X_ridge.shape[0]//2

	# Y_ridge = Y_ridge - Y_ridge.mean(0)
	# X_ridge = X_ridge - X_ridge.mean(0) # This step is questionable

	X_ridge_train = X_ridge[:N_train, :]
	Y_ridge_train = Y_ridge[:N_train, :]

	X_ridge_test = X_ridge[N_train:, :]
	Y_ridge_test = Y_ridge[N_train:, :]

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Choose appropriate regularization parameter using 
	# split-half  cross-validation.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	beta0 = X_ridge_train[:, -1].mean()

	S = numpy.dot(Y_ridge_train.T, Y_ridge_train)

	Ytx = numpy.dot(Y_ridge_train.T, X_ridge_train[:, -1])

	lams = numpy.logspace(-4, 5, 50)

	beta_by_lams = numpy.zeros((N_res, len(lams)))
	err_by_lams = numpy.zeros(len(lams))

	I = numpy.identity(N_res)

	for lam_ind, lam in enumerate(lams):
		if is_verbose:
			print("On lam_ind = {} of {}...".format(lam_ind + 1, len(lams)))

		beta = numpy.linalg.solve(S + lam*I, Ytx)
		beta_by_lams[:, lam_ind] = numpy.ravel(beta)

		Xhat = numpy.dot(Y_ridge_test, beta)

		err_by_lams[lam_ind] = numpy.mean(numpy.power(X_ridge_test[:, -1] - Xhat, 2))

	lam_argmin = numpy.argmin(err_by_lams)
	lam_min = lams[lam_argmin]

	if to_plot:
		plt.figure()
		plt.plot(lams, err_by_lams)
		plt.xscale('log')

		plt.axvline(lam_min, color = 'red')

	if is_verbose:
		print("Split-half CV chose lambda = {}".format(lam_min))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Recompute beta using full time series:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	S = numpy.dot(Y_ridge.T, Y_ridge)

	Ytx = numpy.dot(Y_ridge.T, X_ridge[:, -1])

	beta = numpy.ravel(numpy.linalg.solve(S + lam_min*I, Ytx))

	beta0 = X_ridge[:, -1].mean()

	if is_verbose:
		print("The intercept assuming mean-centered predictors is:\n{}".format(beta0))

	beta0 = beta0 - numpy.sum(beta*numpy.ravel(Y_ridge.mean(0)))

	if is_verbose:
		print("The intercept without  mean-centered predictors is:\n{}".format(beta0))

	Wout_cv = numpy.concatenate(([beta0], beta))

	x_esn = numpy.ravel(numpy.dot(Y_ridge, beta) + beta0)
	err_esn = numpy.ravel(X_ridge[:, -1]) - x_esn

	return Wout_cv, x_esn, err_esn

def estimate_ridge_regression_joint_w_splithalf_cv(X_ridge, Y_ridge, to_plot = False, is_verbose = False):
	N_res = Y_ridge.shape[1]

	N_train = X_ridge.shape[0]//2

	# Y_ridge = Y_ridge - Y_ridge.mean(0)
	# X_ridge = X_ridge - X_ridge.mean(0) # This step is questionable

	X_ridge_train = X_ridge[:N_train, :]
	Y_ridge_train = Y_ridge[:N_train, :]

	X_ridge_test = X_ridge[N_train:, :]
	Y_ridge_test = Y_ridge[N_train:, :]

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Choose appropriate regularization parameter using 
	# split-half  cross-validation.
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	beta0 = X_ridge_train.mean(0)

	S = numpy.dot(Y_ridge_train.T, Y_ridge_train)

	Ytx = numpy.dot(Y_ridge_train.T, X_ridge_train)

	lams = numpy.logspace(-4, 5, 50)

	beta_by_lams = numpy.zeros((N_res, 2, len(lams)))
	err_by_lams = numpy.zeros(len(lams))

	I = numpy.identity(N_res)

	for lam_ind, lam in enumerate(lams):
		if is_verbose:
			print("On lam_ind = {} of {}...".format(lam_ind + 1, len(lams)))

		beta = numpy.linalg.solve(S + lam*I, Ytx)
		beta_by_lams[:, :, lam_ind] = beta

		Xhat = numpy.dot(Y_ridge_test, beta)

		err_by_lams[lam_ind] = numpy.mean(numpy.power(X_ridge_test - Xhat, 2))

	lam_argmin = numpy.argmin(err_by_lams)
	lam_min = lams[lam_argmin]

	if to_plot:
		plt.figure()
		plt.plot(lams, err_by_lams)
		plt.xscale('log')

		plt.axvline(lam_min, color = 'red')

	if is_verbose:
		print("Split-half CV chose lambda = {}".format(lam_min))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#
	# Recompute beta using full time series:
	#
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	S = numpy.dot(Y_ridge.T, Y_ridge)

	Ytx = numpy.dot(Y_ridge.T, X_ridge)

	beta = numpy.linalg.solve(S + lam_min*I, Ytx)

	beta0 = X_ridge.mean(0)

	if is_verbose:
		print("The intercept assuming mean-centered predictors is:\n{}".format(beta0))

	beta0 = beta0 - numpy.dot(Y_ridge.mean(0), beta)

	if is_verbose:
		print("The intercept without  mean-centered predictors is:\n{}".format(beta0))

	Wout_cv = numpy.row_stack((beta0, beta))

	z_esn = numpy.dot(Y_ridge, beta) + beta0

	y_esn = numpy.ravel(z_esn[:, 0])
	x_esn = numpy.ravel(z_esn[:, 1])
	err_esn_y = numpy.ravel(X_ridge[:, 0]) - y_esn
	err_esn_x = numpy.ravel(X_ridge[:, 1]) - x_esn

	return Wout_cv, y_esn, x_esn, err_esn_y, err_esn_x
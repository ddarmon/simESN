import numpy
import ipdb

import matplotlib.pyplot as plt
plt.ion()

# from matplotlib import rc
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('font', size=14)
# plt.rc('figure', figsize=[10,5])

from sklearn.linear_model import Ridge

import getpass

username = getpass.getuser()

import sys

platform = sys.platform

sys.path.append('../../../sidpy/sidpy'.format(username))

sys.path.append('../')

sys.path.append('../simESN'.format(username))

import simESN

import sidpy
import load_models

import scipy

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Demonstrate how scipy.sparse.random behaves:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# N_res = 1000

# # mean_degree = 3
# mean_degree = 10
# p_erdosrenyi = mean_degree/float(N_res)

# W = scipy.sparse.random(m = N_res, n = N_res, density = p_erdosrenyi, data_rvs = scipy.stats.uniform(loc = -0.5, scale = 1).rvs)

# s = scipy.sparse.linalg.svds(W, k=1)

# W = W.multiply(numpy.abs(1/float(s[1])))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Simulate from a model system:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Simulating from model system...")

N = 1000

x = numpy.zeros(N)

# is_stochastic = True
is_stochastic = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Set various parameters:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p_max_ms = 10

# Parameters used for all simulations so far:
# marginal_estimation_procedure = 'knn'
# pow_upperbound = 0.5
# N_res = 400
# rho = 0.99

# Parameters suggested for TTE example: 	
marginal_estimation_procedure = 'knn'
pow_upperbound = 0.5
N_res = 100

# Has ESP:
# rho = 0.1
rho = 0.99

# Intermediate values:
# rho = 1.1
# rho = 2.5

# Does not have ESP:
# rho = 5
# rho = 10 

Win_scale = 1.
# Win_scale = 100.

# p_opt, nlpl_opt, nlpl_by_p, er_knn, ler_knn = sidpy.choose_model_order_nlpl(x, p_max_ms, pow_upperbound=pow_upperbound, marginal_estimation_procedure=marginal_estimation_procedure, nn_package='sklearn', is_multirealization=False, announce_stages=False, output_verbose=True, suppress_warning=False)
# p_opt, nlpl_opt, nlpl_by_p, er_knn = sidpy.choose_model_order_mse(x, p_max_ms, pow_upperbound=pow_upperbound, nn_package='sklearn', is_multirealization=False, announce_stages=False, output_verbose=True)

p_opt = 4
# p_max = p_opt

x_esn0, X0, Y0, err_esn0, Win, W, Wout, bias_constant = simESN.learn_esn_umd_sparse(x, p_max = p_opt, N_res = N_res, rho = rho, Win_scale = Win_scale, output_verbose = False)

x_esn1, X1, Y1, err_esn1, Win, W, Wout, bias_constant = simESN.learn_esn_umd_sparse(x, p_max = p_opt, N_res = N_res, rho = rho, Win_scale = Win_scale, output_verbose = False, Win = Win, bias_constant = bias_constant, W = W, seed_for_ic = 1)

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
cf0 = ax[0].imshow(Y0, aspect = 10.)
cf1 = ax[1].imshow(Y1, aspect = 10.)
fig.colorbar(cf0, ax=ax[0])
fig.colorbar(cf1, ax=ax[1])

# to_plot = 25 # For when has ESP.

to_plot = 1000 # For when does not have ESP.

abs_diff_because_ic = numpy.abs(Y0[:, :to_plot] - Y1[:, :to_plot])
log_abs_diff_because_ic = numpy.log10(abs_diff_because_ic)

plt.figure()
plt.imshow(log_abs_diff_because_ic, aspect = 10)
plt.colorbar()

fig, ax = plt.subplots(5, 1, sharex = True, sharey = True, figsize = (20, 5))
for ax_ind in range(5):
	ax[ax_ind].plot(numpy.ravel(Y0[ax_ind, :]))
	ax[ax_ind].plot(numpy.ravel(Y1[ax_ind, :]))
	ax[ax_ind].set_xlim([0, 100])
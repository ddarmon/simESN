import numpy

import matplotlib.pyplot as plt
plt.ion()

import sys

sys.path.append('../')

import simESN
import sidpy

import load_models

import scipy

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Simulate from a model system:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Simulating from model system...")

N = 1000

# model_name = 'lorenz'
# model_name = 'rossler'
# model_name = 'tent'
# model_name = 'logistic'

model_name = 'slorenz'
# model_name = 'srossler'
# model_name = 'setar'
# model_name = 'stent'
# model_name = 'slogistic'
# model_name = 'arch'

# model_name = 'slogistic'
# model_name = 'setar'
# model_name = 'slorenz'

x, p_true, model_type = load_models.load_model_data(model_name, N)

if model_name == 'shadow_crash':
	x = numpy.log(x)

x_std = numpy.std(x)

x = (x-numpy.mean(x))/x_std

is_stochastic = True
# is_stochastic = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Set various parameters:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Parameters used for all simulations so far:
marginal_estimation_procedure = 'knn'
pow_upperbound = 0.5
N_res = 400
rho = 0.99

Win_scale = 1.

p_opt = 4

# renormalize_by = 'svd'
renormalize_by = 'eigen'

print("NOTE: Using renormalize_by = \'{}\'".format(renormalize_by))

p_opt = 4

print("Learning ESN for time series:")

p_max = p_opt

# multi_bias = False
multi_bias = True; print("NOTE: Using multibias = True.")

x_esn, X, Y, err_esn, Win, W, Wout, bias_constant = simESN.learn_esn_umd_sparse(x, p_max = p_opt, N_res = N_res, rho = rho, Win_scale = Win_scale, multi_bias = True, to_plot_regularization = True, renormalize_by = renormalize_by)

knn_errs = False; print("NOTE: Using knn_errs = False.")
# knn_errs = True; print("NOTE: Using knn_errs = True.")

nn_number = None

X = sidpy.embed_ts(x, p_max = p_opt)

vec_ones = numpy.ones(X.shape[1]).reshape(1, -1)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Continue running the time series forward in
# time 'unlinked' to the original time series.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_sim = N

x_esn_sim = simESN.simulate_from_esn_umd_sparse(N_sim, X.T, Y, err_esn, Win, W, Wout, bias_constant, p_max = p_opt, is_stochastic = is_stochastic, knn_errs = knn_errs, nn_number = nn_number, print_iter = True)

plt.figure(figsize = (15, 5))
plt.plot(x, label = 'True time series')
plt.plot(x_esn_sim, label = 'Time series simulated from ESN')
plt.legend()

plt.figure(figsize = (20, 5))
plt.plot(x, label = 'True time series')
plt.plot(numpy.arange(x_esn_sim.shape[0]) + x.shape[0] - p_max, x_esn_sim, label = 'Time series simulated from ESN')
plt.ylim((numpy.min(x), numpy.max(x)))
plt.legend()

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10*2, 5*2))
ax[0].scatter(x[:-1], x[1:], s = 0.5, label = 'True time series')
ax[0].legend()
ax[1].scatter(x_esn_sim[:-1], x_esn_sim[1:], s = 0.5, label = 'Time series from ESN')
ax[1].legend()
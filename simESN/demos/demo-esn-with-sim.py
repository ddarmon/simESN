import numpy
import ipdb

import matplotlib.pyplot as plt
plt.ion()

# from matplotlib import rc
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('font', size=14)
# plt.rc('figure', figsize=[10,5])

import getpass

username = getpass.getuser()

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
# model_name = 'snanopore'
# model_name = 'shadow_crash'
# model_name = 'setar'
# model_name = 'stent'
# model_name = 'slogistic'
# model_name = 'arch'

# model_name = 'slogistic'
# model_name = 'setar'
# model_name = 'slorenz'

x, p_true, model_type = load_models.load_model_data(model_name, N)

# model_name = 'shenon'; dim = None
# model_name = 'slorenz96'; dim = 47

# y, x, p_true, model_type = load_models.load_model_data_io(model_name, N, dim = dim, ds_by = None)
# x = y

if model_name == 'shadow_crash':
	x = numpy.log(x)

x_std = numpy.std(x)

x = (x-numpy.mean(x))/x_std

is_stochastic = True
# is_stochastic = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Investigate the ISI data:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# model_name = 'isi_data'

# # C = control. P = penicillin
# condition_type = 'P'
# # condition_type = 'C'

# neuron_ind = 2

# x = numpy.loadtxt('/Users/{}/Dropbox (Personal)/Reference/T/tirp/2018/isi-neurons/data/{}{}.DAT'.format(username, condition_type, neuron_ind))
# x = x + (1-0.5*numpy.random.rand(x.shape[0]))*1e-3
# x = numpy.log(x)

# x_std = numpy.std(x)

# x = (x-numpy.mean(x))/x_std

# N = x.shape[0]

# is_stochastic = True
# # is_stochastic = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Investigate synthetic IEI data:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# # model_name = 'lorenz-ibi-thresh60'
# model_name = 'rossler-ibi-thresh125'

# x = numpy.loadtxt('/Users/{}/Google Drive/Reference/T/tirp/2015/hrv-analysis/data/{}.csv'.format(username, model_name), skiprows = 1)

# x = x + 2*(numpy.random.rand(x.shape[0]) - 0.5)*(10**(-2))

# x = numpy.log(x)

# x_std = numpy.std(x)

# x = (x-numpy.mean(x))/x_std

# N = x.shape[0]

# is_stochastic = True
# # is_stochastic = False

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
N_res = 400


rho = 0.99 # Has ESP.

Win_scale = 1.
# Win_scale = 100.

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
# nn_number = int(numpy.power(x.shape[0] - p_opt, 4./(p_opt+1 + 4.))); print("NOTE: Using default nn_number = {}.".format(nn_number)) # MSE optimal for knn density estimation
# nn_number = int(numpy.power(x.shape[0] - p_opt, 2./(p_opt+1 + 2.))); print("NOTE: Using default nn_number = {}.".format(nn_number)) # MSE optimal for regression
# nn_number = 100; print("NOTE: Using nn_number = {}.".format(nn_number))

X = sidpy.embed_ts(x, p_max = p_opt)

vec_ones = numpy.ones(X.shape[1]).reshape(1, -1)

#
####################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Use a LAR model that does not incorporate the 
# reservoir states:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S_lm = numpy.row_stack((vec_ones, X[:-1, :]))
Wout_lm = numpy.linalg.solve(numpy.dot(S_lm.T, S_lm), numpy.dot(S_lm.T, X[:, -1]))

Xhat_lm = numpy.dot(S_lm, Wout_lm)

x_lm = numpy.ravel(Xhat_lm)

err_lm = x[p_max:] - x_lm

fig, ax = plt.subplots(2, sharex = True)
ax[0].plot(x[p_max:], label = 'True time series')
ax[0].plot(x_esn, label = 'Predicted using ESN')
ax[0].plot(x_lm, label = 'Predicted using LM')
ax[0].set_ylabel('$X_{t}$')
ax[0].legend()
ax[1].plot(0)
ax[1].plot(err_esn)
ax[1].set_ylabel('$X_{t} - \widehat{X}_{t, ESN}$')
# ax[1].plot(err_lm)

mse_esn = numpy.power(err_esn, 2.).mean()
mse_lm = numpy.power(err_lm, 2.).mean()

print("MSE(ESN) = {}\nMSE(LM ) = {}".format(mse_esn, mse_lm))

fig, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (15*1.5, 5*1.5))

ax[0].scatter(x[:-1], x[1:], s = 0.5, color = 'blue')
ax[1].scatter(x_esn[:-1], x_esn[1:], s = 0.5, color = 'orange')
ax[2].scatter(x_lm[:-1], x_lm[1:], s = 0.5, color = 'green')

if p_opt == 1:
	fig, ax = plt.subplots(p_opt, 1)
	p_plot = 1
	ax.scatter(X[:, -(p_plot + 1)], x_esn, s = 0.5)
	ax.set_xlabel('$X_{{t-{}}}$'.format(p_plot))
	ax.set_ylabel('Pred$(X_{{t}})$'.format(p_plot))

	fig, ax = plt.subplots(p_opt + 1, 1)
	ax[0].scatter(err_esn[:-1], err_esn[1:], s = 0.5)
	ax[0].set_xlabel('$\eta_{{t}}$')
	ax[0].set_ylabel('$\eta_{{t+1}}$')
	ax[0].legend()
	for p_plot in range(1, p_opt+1):
		ax[p_plot].scatter(X[:, -(p_plot + 1)], err_esn, s = 0.5)
		ax[p_plot].set_xlabel('$X_{{t-{}}}$'.format(p_plot))
		ax[p_plot].set_ylabel('$\eta_{{t}}$'.format(p_plot))
else:
	fig, ax = plt.subplots(p_opt, 1)
	for p_plot in range(1, p_opt+1):
		ax[p_plot-1].scatter(X[:, -(p_plot + 1)], x_esn, s = 0.5)
		ax[p_plot-1].set_xlabel('$X_{{t-{}}}$'.format(p_plot))
		ax[p_plot-1].set_ylabel('Pred$(X_{{t}})$'.format(p_plot))

	fig, ax = plt.subplots(p_opt + 1, 1)
	ax[0].scatter(err_esn[:-1], err_esn[1:], s = 0.5)
	ax[0].set_xlabel('$\eta_{{t}}$')
	ax[0].set_ylabel('$\eta_{{t+1}}$')
	ax[0].legend()
	for p_plot in range(1, p_opt+1):
		ax[p_plot].scatter(X[:, -(p_plot + 1)], err_esn, s = 0.5)
		ax[p_plot].set_xlabel('$X_{{t-{}}}$'.format(p_plot))
		ax[p_plot].set_ylabel('$\eta_{{t}}$'.format(p_plot))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Continue running the time series forward in
# time 'unlinked' to the original time series.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_sim = N

# if 'eeg' in model_name:
# 	N_sim = 1000
# else:
# 	N_sim = N

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Simulate a very long realization.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# N_sim = N*10
N_sim = 10000

x_esn_sim_verylong = simESN.simulate_from_esn_umd_sparse(N_sim, X.T, Y, err_esn, Win, W, Wout, bias_constant, p_max = p_opt, is_stochastic = is_stochastic, knn_errs = knn_errs, nn_number = nn_number, print_iter = True)

plt.figure(figsize = (15, 5))
plt.plot(x, label = 'True time series')
plt.plot(x_esn_sim_verylong, label = 'Time series from ESN')
plt.legend()

plt.figure(figsize = (20, 5))
plt.plot(x)
plt.plot(numpy.arange(x_esn_sim_verylong.shape[0]) + x.shape[0] - p_max, x_esn_sim_verylong)
plt.ylim((numpy.min(x), numpy.max(x)))

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, figsize = (10*2, 5*2))
ax[0].scatter(x[:-1], x[1:], s = 0.5, label = 'True time series')
ax[0].legend()
ax[1].scatter(x_esn_sim_verylong[:-1], x_esn_sim_verylong[1:], s = 0.5, label = 'Time series from ESN')
ax[1].legend()

plt.show()
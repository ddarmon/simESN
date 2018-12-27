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

import sys

sys.path.append('../')

import simESN

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
N_res = 100

# rho = 0.1 # Has ESP
rho = 0.99 # Has ESP.
# rho = 2.0 # Has ESP.

# rho = 4.0 # Has ESP after long time.

# rho = 8.0 # Does not have ESP.
# rho = 10 # Does not have ESP.

Win_scale = 1.
# Win_scale = 100.

p_opt = 4

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Create two versions of the *same* ESN
# architecture, where the only difference is the 
# initial state of the echo state nodes.
#
# Show that, despite this, the nodes of the 
# echo state network still synchronize to the same
# state when provided with the same input signal.
# 
# Note: Change rho above to either:
# 	* 'break', for large enough rho
# 	* maintain, for small enough rho
# 
# the echo state property for this network.
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_esn0, X0, Y0, err_esn0, Win, W, Wout, bias_constant = simESN.learn_esn_umd_sparse(x, p_max = p_opt, N_res = N_res, rho = rho, Win_scale = Win_scale, output_verbose = False)

x_esn1, X1, Y1, err_esn1, Win, W, Wout, bias_constant = simESN.learn_esn_umd_sparse(x, p_max = p_opt, N_res = N_res, rho = rho, Win_scale = Win_scale, output_verbose = False, Win = Win, bias_constant = bias_constant, W = W, seed_for_ic = 1)

fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
im0 = ax[0].imshow(Y0, aspect = 10.)
im1 = ax[1].imshow(Y1, aspect = 10.)
fig.colorbar(im0, ax = ax[0])
fig.colorbar(im1, ax = ax[1])

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
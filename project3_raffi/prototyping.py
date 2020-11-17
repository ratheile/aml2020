# %%
# should provide auto reload
# %load_ext autoreload
# %autoreload 2
# Path hack.import sys, os

# %%
sys.path.insert(0, os.path.abspath('..'))

from pywt import wavedec, waverec

import heartpy as hp
import neurokit2 as nk

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from copy import deepcopy
from modules import ConfigLoader
from project2_raffi.visualization import pca


import plotly.express as px

# %% some local testing:
env_cfg = ConfigLoader().from_file('../env/env.yml')
# %%
print(env_cfg)

# %%
df_X = pd.read_csv(f"{env_cfg['datasets/project3/path']}/X_train.csv")
df_y = pd.read_csv(f"{env_cfg['datasets/project3/path']}/y_train.csv")
df_X_u = pd.read_csv(f"{env_cfg['datasets/project3/path']}/X_test.csv")  # unlabeled

#%%
sample_rate = 300
def time_scale(samples): return np.arange(samples.shape[0]) / sample_rate

# %%
X = df_X.iloc[:, 1:]
y = df_y.iloc[:, 1:].values.ravel()
X_u = df_X.iloc[:, 1:]

X_124 = X.iloc[y != 3]
X_3 = X.iloc[y == 3]


#%%
crv = X_124.iloc[11, ]
crv = X_3.iloc[24]
crv = crv[~np.isnan(crv.values.astype(float))]
crv_df = pd.DataFrame({
    'v': crv.values,
    't': time_scale(crv)
})

px.line(crv_df, x='t', y='v')


# %%
def wavelet_decomposition(sig):
    cA5, cD5, cD4, cD3, cD2, cD1 = wavedec(sig, 'bior4.4', level=5)
    coeffs = {'cA5': cA5, 'cD5': cD5, 'cD4': cD4,
              'cD3': cD3, 'cD2': cD2, 'cD1': cD1}
    return coeffs


def wavelet_reconstruction(coeffs):
    reconstructed = waverec([coeffs['cA5'], coeffs['cD5'], coeffs['cD4'], coeffs['cD3'],
                             coeffs['cD2'], coeffs['cD1']], 'bior4.4')
    return reconstructed


# define the threshold percentage for retaining energy of wavelet coefficients
# separate percentage for approximate coefficients and separate for detailed
THRESH_PERC_APPROX = 0.999
THRESH_PERC_D5 = 0.97
THRESH_PERC_D4_D1 = 0.5

# don't allow the PRD to be greater than 5%
MAX_PRD = 0.4


def energy(sig):
    return np.sum(sig**2)


def calculate_PRD(orig_sig, reconstructed_sig):
    num = np.sum((orig_sig - reconstructed_sig)**2)
    den = np.sum(orig_sig**2)

    PRD = np.sqrt(num/den)

    return PRD


def threshold_energy(coeffs):
    #make a deep copy of coeffs to retain the original version
    coeffs_orig = deepcopy(coeffs)

    binary_map = {}
    nonzero_coeff_count = {}

    for key in coeffs.keys():
        #sort the absolute value of the coefficients in descending order
        tmp_coeffs = np.sort(np.abs(coeffs[key]))[::-1]

        #calculate the threshold for retaining some percentage of the energy
        if key == 'cA5':
            thresh_perc = THRESH_PERC_APPROX
        elif key == 'cD5':
            thresh_perc = THRESH_PERC_D5
        else:
            thresh_perc = THRESH_PERC_D4_D1

        energy_thresholded = thresh_perc*energy(tmp_coeffs)
        energy_tmp = 0
        for coeff in tmp_coeffs:
            energy_tmp = energy_tmp + coeff**2

            if energy_tmp >= energy_thresholded:
                threshold = coeff
                break

        #set any coefficients below the threshold to zero
        tmp_coeffs = coeffs[key]
        inds_to_zero = np.where((tmp_coeffs < threshold) & (tmp_coeffs > -threshold))[0]
        tmp_coeffs[inds_to_zero] = 0

        #create the binary map
        binary_map_tmp = np.ones(len(coeffs[key])).astype(int)
        binary_map_tmp[inds_to_zero] = 0

        #update the various dictionaries
        coeffs[key] = tmp_coeffs
        binary_map[key] = binary_map_tmp
        nonzero_coeff_count[key] = len(tmp_coeffs)

    return coeffs, binary_map


def do_quantization(coeffs, bits):
    quantized_coeffs = {}

    for key in coeffs.keys():
        sig = coeffs[key]
        sig = sig*(2**bits-1)
        sig = np.round(sig)
        sig = np.array(sig).astype(int)

        quantized_coeffs[key] = sig
    return quantized_coeffs

def scale_coeffs(coeffs):
    coeffs_scaled = {}
    scaling_factors = {}

    for key in coeffs.keys():fig1.set_size_inches(18.5, 10.5)
        shift_factor = np.min(coeffs[key])
        coeffs_tmp = coeffs[key]-shift_factor

        scale_factor = np.max(coeffs_tmp)
        coeffs_tmp = coeffs_tmp/scale_factor

        scaling_factors[key] = {
            'shift_factor': shift_factor, 'scale_factor': scale_factor}
        coeffs_scaled[key] = coeffs_tmp

    return coeffs_scaled, scaling_factors


def unscale_coeffs(coeffs, scaling_factors, bits):
    coeffs_unscaled = {}

    for key in coeffs.keys():
        tmp_coeffs_unscaled = coeffs[key]/(2**bits)
        tmp_coeffs_unscaled = tmp_coeffs_unscaled * \
            scaling_factors[key]['scale_factor']
        tmp_coeffs_unscaled = tmp_coeffs_unscaled + \
            scaling_factors[key]['shift_factor']

        # now replace the NaN values with 0
        nan_inds = np.where(np.isnan(tmp_coeffs_unscaled))[0]
        tmp_coeffs_unscaled[nan_inds] = 0

        coeffs_unscaled[key] = tmp_coeffs_unscaled

    return coeffs_unscaled


def calculate_num_bits(orig_sig, coeffs_scaled, binary_map, scaling_factors):
    # starting at 8 bits, keep decreasing the number of bits in the quantization
    # until the PRD is above some threshold
    num_bits = 9

    # initialize PRD to 0 so the while loop can run
    PRD = 0

    # keep track of PRD per number of bits
    PRD_dict = {}

    while (num_bits >= 5) and (PRD <= MAX_PRD):
        # decrement the number of bits
        num_bits = num_bits-1

        coeffs_quantized = do_quantization(coeffs_scaled, num_bits)

        # rescale the coefficients
        coeffs_unscaled = unscale_coeffs(
            coeffs_quantized, scaling_factors, num_bits)

        # do the inverse dwt
        data_reconstructed = wavelet_reconstruction(coeffs_unscaled)

        # calculate PRD
        PRD = calculate_PRD(orig_sig, data_reconstructed)
        PRD_dict[num_bits] = PRD

    # if we went over the PRD, go back up by one bit
    if PRD > MAX_PRD:
        num_bits = num_bits+1
        PRD = PRD_dict[num_bits]
      
    return num_bits, PRD



#%%

crv = X_124.iloc[11, ]
crv = X_3.iloc[24]
crv = crv[~np.isnan(crv.values.astype(float))]

#%%
# coeffs = wavelet_decomposition(crv)
# f, axes = plt.subplots(nrows=len(coeffs), ncols=1, figsize=(24, 12))
# for i, item in enumerate(coeffs.items()):
#     k, v = item
#     ax = axes[i]
#     ax.plot(v)
#     ax.set_title(k)

# coeffs_thds, binary_map = threshold_energy(coeffs)
# coeffs_scaled, scaling = scale_coeffs(coeffs_thds)
# num_bits, PRD_tmp = calculate_num_bits(crv, coeffs_scaled, binary_map, scaling)

# coeffs_quantized = do_quantization(coeffs_scaled, num_bits)
# coeffs_unscaled = unscale_coeffs(coeffs_quantized, scaling, num_bits)

# # rec = wavelet_reconstruction(coeffs_unscaled)
# # f, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 12))
# # ax.plot(time_scale(rec), rec, label='Original Signal')
# # ax.plot(time_scale(crv), crv, label='Reconstructed Signal')
# # f.tight_layout()

# # %%
# f, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 12))
# ax.plot(list(range(len(crv))), crv)



# Automatically process the (raw) ECG signal
ecg_orig_clean = nk.ecg_clean(crv.values.astype(float))
signals, info = nk.ecg_process(ecg_orig_clean, sampling_rate=sample_rate)

#%%
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 12))
ax.plot(list(range(len(ecg_orig_clean))), ecg_orig_clean)
# %%

# %%
# run analysis
wd, m = hp.process(ecg_orig_clean, sample_rate)
# wd2, m2 = hp.process(crv, sample_rate)

# visualise in plot of custom size
plt.figure(figsize=(12, 4))
hp.plotter(wd, m)

# display computed measures
for measure in m.keys():
    print('%s: %f' % (measure, m[measure]))
# %%
# %%
fig1 = nk.ecg_plot(signals, sampling_rate=sample_rate, show_type='default') 
fig1.set_size_inches(18.5, 10.5)
fig2 = nk.ecg_plot(signals, sampling_rate=sample_rate, show_type='artifacts') 

# %%
# Find peaks
peaks, info = nk.ecg_peaks(ecg_orig_clean, sampling_rate=sample_rate)

# Compute HRV indices
fig1 = nk.hrv(peaks, sampling_rate=sample_rate, show=True)
# %%

# %%

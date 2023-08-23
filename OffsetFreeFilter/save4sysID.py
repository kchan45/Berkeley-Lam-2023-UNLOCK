import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

save_file = True
filepath = './ExperimentalData/2023_08_21_17h31m03s/Backup/OL_data_0.npy'
timestamp = filepath[19:39]
print(timestamp)
Fontsize = 14 # default font size for plots
Lwidth = 2 # default line width for plots

## plotting setup
lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

processing_info = {}
processing_info['raw_data_file'] = filepath
processing_info['timestamp'] = timestamp

exp_data = np.load(filepath, allow_pickle=True).item()
print(exp_data.keys())
print(exp_data['badTimes'])

# raw data
Tplot = exp_data['Tsave']
Iplot = exp_data['Isave']
Pplot = exp_data['Psave']
qplot = exp_data['qSave']
specData = exp_data['specSave']
specDataRaw = specData + exp_data['meanShiftSave'].reshape(-1,1)

# subtract a constant mean shift rather than a different one per time
avg_meanShift = np.mean(exp_data['meanShiftSave'])
processing_info['mean_shift'] = avg_meanShift
specData = specDataRaw - avg_meanShift

# grab desired peaks
I706idx = 1014
I777idx = 1226
processing_info['I706idx'] = I706idx
processing_info['I777idx'] = I777idx

# grab background
nbackground = 45
background = np.mean(specData[:nbackground,:],axis=0)
print(background.shape)
processing_info['background'] = background
specData = specData - background

# remove background from data
Tplot = Tplot[nbackground:]
I706plot = specData[nbackground:,I706idx]
I777plot = specData[nbackground:,I777idx]
Pplot = Pplot[nbackground:]
qplot = qplot[nbackground:]

# grab nominal steady state
n_steady_state = int(45*3)
y = np.vstack((Tplot, I706plot, I777plot))
y_min = np.min(y, axis=1).reshape(-1,1)
y_max = np.max(y, axis=1).reshape(-1,1)
y = 2*(y-y_min)/(y_max-y_min) - 1
processing_info['y_min'] = y_min
processing_info['y_max'] = y_max

u = np.vstack((Pplot, qplot))
u_min = np.min(u, axis=1)
u_max = np.max(u, axis=1)

fig, axes = plt.subplots(2,1, figsize=(12,8))
axes[0].plot(y.T)
axes[0].set_title('Outputs')
axes[1].plot(u.T)
axes[1].set_title('Inputs')
plt.tight_layout()

yss = np.mean(y[:, :n_steady_state], axis=1)
print('yss: ', yss)
uss = np.mean(u[:, :n_steady_state], axis=1)
print('uss: ', uss)
processing_info['yss'] = yss
processing_info['uss'] = uss
y = y[:,n_steady_state:] - yss.reshape(-1,1)
print(y.shape)
u = u[:,n_steady_state:] - uss.reshape(-1,1)

processing_info['y'] = y
processing_info['u'] = u

fig, axes = plt.subplots(2,1, figsize=(12,8))
axes[0].plot(y.T)
axes[0].set_title('Outputs')
axes[1].plot(u.T)
axes[1].set_title('Inputs')
plt.tight_layout()

if save_file:
    savemat(f'./models/{timestamp}_APPJ_model_train_data.mat', processing_info)

plt.show()
import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import savemat

save_file = False
filepath = './ExperimentalData/2023_08_21_17h31m03s/Backup/OL_data_0.npy'
timestamp = filepath[19:39]
print(timestamp)
Fontsize = 8 # default font size for plots
Lwidth = 1 # default line width for plots

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
wavelengths = exp_data['waveSave'][1000,20:]

fig, ax = plt.subplots(1,1,dpi=300,figsize=(6,3))
axins = inset_axes(ax, width="15%", height="50%",
                   bbox_to_anchor=(.395, .5, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)
axins2 = inset_axes(ax, width="15%", height="50%",
                   bbox_to_anchor=(.65, .445, .6, .5),
                   bbox_transform=ax.transAxes, loc=3)
mask = (wavelengths >= 704.0) & (wavelengths <= 710.0)
mask2 = (wavelengths >= 774.0) & (wavelengths <= 781.0)

ax.plot(wavelengths, specDataRaw[1000,20:], label='Raw', color='c')

# subtract a constant mean shift rather than a different one per time
avg_meanShift = np.mean(exp_data['meanShiftSave'])
processing_info['mean_shift'] = avg_meanShift
specData = specDataRaw - avg_meanShift

ax.plot(wavelengths, specData[1000,20:], label='Zeroed', color='m', alpha=0.7, ls='--')
axins.plot(wavelengths[mask], specData[1000,20:][mask], color='m', alpha=0.7, ls='--')
axins2.plot(wavelengths[mask2], specData[1000,20:][mask2], color='m', alpha=0.7, ls='--')

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

ax.plot(wavelengths, background[20:], label='Background', color='b')
axins.plot(wavelengths[mask], background[20:][mask], color='b')
axins2.plot(wavelengths[mask2], background[20:][mask2], color='b')

ax.plot(wavelengths, specData[1000,20:], label='Zeroed + Background Removed', alpha=0.7, color='g')
axins.plot(wavelengths[mask], specData[1000,20:][mask], color='g', alpha=0.7)
axins2.plot(wavelengths[mask2], specData[1000,20:][mask2], color='g', alpha=0.7)
axins.set_title('He706 Peak')
axins2.set_title('O777 Peak')
for a in [axins, axins2]:
    (a.title).set_fontsize(7)
    xtick_labels = a.get_xticklabels()
    for item in xtick_labels:
        item.set_fontsize(6)
    ytick_labels = a.get_yticklabels()
    for item in ytick_labels:
        item.set_fontsize(6)

ax.legend(loc='upper right', fontsize='x-small')
ax.set_ylabel('Intensity (arb. units)')
ax.set_xlabel('Wavelength (nm)')
plt.tight_layout()

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

fig, axes = plt.subplots(2,1, figsize=(10,3), sharex=True, dpi=300)
axes[0].plot(y.T)
axes[0].set_title('Outputs')
axes[0].legend(['Surface Temperature ('+r'$^\circ$'+' C)', 'Intensity at He706 Peak (arb. units)', 'Intensity at O777 Peak (arb. units)'], fontsize='x-small')
axes[0].set_ylabel('y')
axes[1].plot(u.T)
axes[1].set_title('Inputs')
axes[1].legend(['Applied Power (W)', 'Helium Flow Rate (SLM)'])
axes[1].set_xlabel('Sampling Step')
axes[1].set_ylabel('u')
fig.suptitle('Before Subtracting Nominal Steady State')
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

fig, axes = plt.subplots(2,1, figsize=(12,3.25), dpi=300)
axes[0].plot(y.T)
axes[0].set_title('Outputs')
axes[0].legend(['Surface Temperature ('+r'$^\circ$'+'C)', 'Intensity at He706 Peak (arb. units)', 'Intensity at O777 Peak (arb. units)'], fontsize='x-small', ncol=3, loc='lower center')
axes[0].set_xlabel('Sampling Step')
axes[0].set_ylabel('y')
axes[0].set_ylim([-2.5, 1.0])
axes[1].plot(u.T)
axes[1].set_title('Inputs')
axes[1].legend(['Applied Power (W)', 'Helium Flow Rate (SLM)'], fontsize='x-small', ncol=2, loc='lower center')
axes[1].set_xlabel('Sampling Step')
axes[1].set_ylabel('u')
axes[1].set_ylim([-4.75, 1.5])
# fig.suptitle('Processed Data for System Identification')
plt.tight_layout()

if save_file:
    savemat(f'./models/{timestamp}_APPJ_model_train_data.mat', processing_info)

plt.show()
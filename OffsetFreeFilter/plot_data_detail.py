# plotting script for viewing closed-loop experimental data

import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

## data processing file
processing_info_file = './models/2023_08_21_17h31m03s_APPJ_model_train_data.mat'
processing_info = loadmat(processing_info_file)

# experiment switch
experiment_name = 'center' # options: ['no filter', 'center', 'off center']
experiment_time = 'afternoon' # options: ['morning', 'afternoon']


font_size = 14  # custom default font size for plots
line_width = 2  # custom default line width for plots

## plotting setup
lines = {'linewidth': line_width}
plt.rc('lines', **lines)
font = {
    'family': 'serif',
    'serif': 'Times',
    'size': font_size,
}
plt.rc('font', **font)

def plot_data(filepaths, mpc_type, processing_info, fig_objs=None):

    I706idx = processing_info['I706idx']
    I777idx = processing_info['I777idx']
    y_min_scale = processing_info['y_min']
    y_max_scale = processing_info['y_max']
    mean_shift = processing_info['mean_shift']
    background = np.ravel(processing_info['background'])
    yss = np.ravel(processing_info['yss'])

    if mpc_type == 'nominal':
        color = 'b'
        label = 'Nominal'
    elif mpc_type == 'offsetfree':
        color = 'g'
        label = 'Offset Free'
    else:
        color = 'k'
        label = 'Unknown MPC'

    exp_datas = [np.load(f, allow_pickle=True).item() for f in filepaths]

    T_datas = np.vstack([exp_data['Tsave'] for exp_data in exp_datas])
    P_datas = np.vstack([exp_data['Psave'] for exp_data in exp_datas])
    q_datas = np.vstack([exp_data['qSave'] for exp_data in exp_datas])
    Yref = [exp_data['Yrefsim'] for exp_data in exp_datas][0]
    Tref = ((Yref[0,:] + yss[0]) + 1)/2 * (y_max_scale[0]-y_min_scale[0]) + y_min_scale[0]
    I706ref = np.ravel(((Yref[1,:] + yss[1]) + 1)/2 * (y_max_scale[1]-y_min_scale[1]) + y_min_scale[1] + background[I706idx] + mean_shift)

    spec_datas = [exp_data['specSave'] for exp_data in exp_datas]
    I706_datas = np.vstack([np.ravel(specData[:,I706idx]) for specData in spec_datas])
    I777_datas = np.vstack([np.ravel(specData[:,I777idx]) for specData in spec_datas])
    Ymeas_datas = [exp_data['Ymeas'] for exp_data in exp_datas]
    I706_datas = np.vstack([Ymeas[1,:-1] for Ymeas in Ymeas_datas])
    I777_datas = np.vstack([Ymeas[2,:-1] for Ymeas in Ymeas_datas])

    meanT = np.ravel(np.mean(T_datas, axis=0))
    meanI706 = np.ravel(np.mean(I706_datas, axis=0))
    meanI777 = np.ravel(np.mean(I777_datas, axis=0))
    meanP = np.ravel(np.mean(P_datas, axis=0))
    meanq = np.ravel(np.mean(q_datas, axis=0))

    stdT = np.ravel(np.std(T_datas, axis=0))
    stdI706 = np.ravel(np.std(I706_datas, axis=0))
    stdI777 = np.ravel(np.std(I777_datas, axis=0))
    stdP = np.ravel(np.std(P_datas, axis=0))
    stdq = np.ravel(np.std(q_datas, axis=0))

    first_plot = False
    if fig_objs is None:
        first_plot = True
        fig, axes = plt.subplots(3,2, figsize=(12,10), sharex=True)
        fig_objs = {}
        fig_objs['fig'] = fig
        fig_objs['axes'] = axes
    else:
        fig = fig_objs['fig']
        axes = fig_objs['axes']

    t = np.arange(meanT.shape[0])
    if first_plot:
        axes[0,0].plot(t, Tref, color='k', ls='--', label='Setpoint')
    axes[0,0].plot(t, meanT, color=color, label=label, alpha=0.8)
    axes[0,0].fill_between(
        t, 
        meanT-2*stdT, 
        meanT+2*stdT, 
        alpha=0.1,
        color=color,
    )
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Surface Temperature\n('+r'$^\circ$'+' C)')
    axes[0,0].legend()

    if first_plot:
        axes[0,1].plot(t, I706ref, color='k', ls='--', label='Setpoint')
    axes[0,1].plot(t, meanI706, color=color, label=label, alpha=0.8)
    axes[0,1].fill_between(
        t, 
        meanI706-2*stdI706, 
        meanI706+2*stdI706, 
        alpha=0.1,
        color=color,
    )
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Intensity at He706\nPeak (arb. units)')

    axes[1,0].plot(t, meanI777, color=color, label=label)
    axes[1,0].fill_between(
        t, 
        meanI777-2*stdI777, 
        meanI777+2*stdI777, 
        alpha=0.1,
        color=color,
    )
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Intensity at He777\nPeak (arb. units)')

    if first_plot:
        axes[2,0].axhline(1.5, color='r', ls='--')
        axes[2,0].text(0.9, 0.05, 'Minimum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,0].transAxes,
               color='r',
               )
        axes[2,0].axhline(3.5, color='r', ls='--')
        axes[2,0].text(0.9, 0.95, 'Maximum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,0].transAxes,
               color='r',
               )
        
        axes[2,1].axhline(1.5, color='r', ls='--')
        axes[2,1].text(0.9, 0.05, 'Minimum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,1].transAxes,
               color='r',
               )
        axes[2,1].axhline(5.5, color='r', ls='--')
        axes[2,1].text(0.9, 0.95, 'Maximum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,1].transAxes,
               color='r',
               )

    for Pdata in P_datas:
        axes[2,0].step(t, Pdata, color=color, label=label, alpha=0.3)
    # axes[2,0].step(t, meanP, color=color, label=label)
    # axes[2,0].fill_between(
    #     t, 
    #     meanP-2*stdP, 
    #     meanP+2*stdP, 
    #     alpha=0.1,
    #     color=color,
    # )
    axes[2,0].set_ylim(1.25, 3.75)
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_ylabel('Power (W)')

    for qdata in q_datas:
        axes[2,1].step(t, qdata, color=color, label=label, alpha=0.3)
    # axes[2,1].step(t, meanq, color=color, label=label)
    # axes[2,1].fill_between(
    #     t, 
    #     meanq-2*stdq, 
    #     meanq+2*stdq, 
    #     alpha=0.1,
    #     color=color,
    # )
    axes[2,1].set_ylim(1.0, 6.0)
    axes[2,1].set_xlabel('Time (s)')
    axes[2,1].set_ylabel('Helium Flow Rate\n(SLM)')
    plt.draw()

    return fig_objs


# no filter
if experiment_name == 'no filter':
    OD_value = 'no filter'
    if experiment_time == 'morning':
        # morning
        nominal_timestamps = [
            '2023_08_23_10h05m43s',
            '2023_08_23_10h10m23s',
            '2023_08_23_10h15m04s',
        ]
        offset_timestamps = [
            '2023_08_23_10h21m22s',
            '2023_08_23_10h26m01s',
            '2023_08_23_10h32m05s',
        ]
    elif experiment_time == 'afternoon':
        # afternoon
        nominal_timestamps = [
            '2023_08_23_15h45m53s',
            '2023_08_23_15h50m50s',
            '2023_08_23_15h55m48s',
        ]
        offset_timestamps = [
            '2023_08_23_16h01m22s',
            '2023_08_23_16h05m58s',
            '2023_08_23_16h11m00s',
        ]

# center
elif experiment_name == 'center':
    OD_value = 'center'
    # morning
    if experiment_time == 'morning':
        nominal_timestamps = [
            '2023_08_23_10h52m51s',
            '2023_08_23_10h57m44s',
            '2023_08_23_11h02m24s',
        ]
        offset_timestamps = [
            '2023_08_23_11h08m47s',
            '2023_08_23_11h13m25s',
            '2023_08_23_11h18m06s',
        ]

    # afternoon
    elif experiment_time == 'afternoon':
        nominal_timestamps = [
            '2023_08_23_16h22m06s',
            '2023_08_23_16h26m47s',
            '2023_08_23_16h32m21s',
        ]
        offset_timestamps = [
            '2023_08_23_16h36m48s',
            '2023_08_23_16h41m20s',
            '2023_08_23_16h46m08s',
        ]

# ~1 cm off center
elif experiment_name == 'off center':
    OD_value = '~1 cm off center'
    # morning
    if experiment_time == 'morning':
        nominal_timestamps = [
            '2023_08_23_11h36m40s',
            '2023_08_23_11h41m19s',
            '2023_08_23_11h45m59s',
        ]
        offset_timestamps = [
            '2023_08_23_11h50m40s',
            '2023_08_23_11h56m39s',
            '2023_08_23_12h01m36s',
        ]

    # afternoon
    elif experiment_time == 'afternoon':
        nominal_timestamps = [
            '2023_08_23_16h59m26s',
            '2023_08_23_17h04m04s',
            '2023_08_23_17h08m37s',
        ]
        offset_timestamps = [
            '2023_08_23_17h14m20s',
            '2023_08_23_17h19m07s',
            '2023_08_23_17h24m26s',
        ]

## Plot Nominal
filepaths = [f'./ExperimentalData/{t}/Backup/Experiment_0.npy' for t in nominal_timestamps]
mpc_type = 'nominal'
fig_objs = plot_data(filepaths, mpc_type, processing_info)

## Plot Offset-free
filepaths = [f'./ExperimentalData/{t}/Backup/Experiment_0.npy' for t in offset_timestamps]
mpc_type = 'offsetfree'
fig_objs = plot_data(filepaths, mpc_type, processing_info, fig_objs=fig_objs)

fig_objs['fig'].suptitle(f'Experiment: {OD_value}')
plt.tight_layout()

plt.show()
# simple plotting script for verifying data collection

import sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt

filepath = ''
open_loop_data = True

Fontsize = 14 # default font size for plots
Lwidth = 2 # default line width for plots


## plotting setup
lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

exp_data = np.load(filepath, allow_pickle=True).item()

Tplot = exp_data['Tsave']
Iplot = exp_data['Isave']
Pplot = exp_data['Psave']
qplot = exp_data['qSave']
specData = exp_data['specSave']
I706plot = specData[1029,:]
I777plot = specData[1232,:]

if open_loop_data:
    fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True)
    
    axes[0,0].plot(Tplot)
    axes[0,0].set_xlabel('Sampling Iteration')
    axes[0,0].set_ylabel('Max Surface\nTemperature (^\circ C)')
    
    axes[0,1].plot(Iplot)
    axes[0,1].set_xlabel('Sampling Iteration')
    axes[0,1].set_ylabel('Total Intensity\n(arb. units)')
    
    axes[1,0].plot(Pplot)
    axes[1,0].set_xlabel('Sampling Iteration')
    axes[1,0].set_ylabel('Applied Power (W)')
    
    axes[1,1].plot(qplot)
    axes[1,1].set_xlabel('Sampling Iteration')
    axes[1,1].set_ylabel('Helium Flow\n Rate (SLM)')

    plt.tight_layout()
    plt.draw()

    fig2, axes2 = plt.subplots(2,1, figsize=(8,4), sharex=True)

    axes2[0,0].plot(I706plot)
    axes2[0,0].set_xlabel('Sampling Iteration')
    axes2[0,0].set_ylabel('Intensity at He706\n(arb. units)')

    axes2[1,0].plot(I777plot)
    axes2[1,0].set_xlabel('Sampling Iteration')
    axes2[1,0].set_ylabel('Intensity at O777\n(arb. units)')

    plt.tight_layout()
    plt.draw()

else:
    Tref = exp_data['Yref'][0,:]
    I706ref = exp_data['Yref'][1,:]
    fig, axes = plt.subplots(3,2, figsize=(12,8), sharex=True)
    
    axes[0,0].plot(Tref, 'k:', label='Setpoint')
    axes[0,0].plot(Tplot)
    axes[0,0].set_xlabel('Sampling Iteration')
    axes[0,0].set_ylabel('Max Surface\nTemperature (^\circ C)')
    axes[0,0].legend()
    
    axes[0,1].plot(I706ref, 'k:', label='Setpoint')
    axes[0,1].plot(I706plot)
    axes[0,1].set_xlabel('Sampling Iteration')
    axes[0,1].set_ylabel('Intensity at He706\n(arb. units)')
    axes[0,1].legend()

    axes[1,0].plot(I777plot)
    axes[1,0].set_xlabel('Sampling Iteration')
    axes[1,0].set_ylabel('Intensity at O777\n(arb. units)')
    
    axes[2,0].plot(Pplot)
    axes[2,0].set_xlabel('Sampling Iteration')
    axes[2,0].set_ylabel('Applied Power (W)')
    
    axes[2,1].plot(qplot)
    axes[2,1].set_xlabel('Sampling Iteration')
    axes[2,1].set_ylabel('Helium Flow\n Rate (SLM)')

    plt.tight_layout()
    plt.draw()


plt.show()

# Import Packages
import numpy as np
import matplotlib.pyplot as plt

# import custom code
from config.my_system import get_prob_info
from utils.controller import OffsetFreeMPC, NominalMPC
from utils.simulation import Simulation
from utils.observer import EKF

### user inputs/options
ts = 1.0 # sampling time, ensure it is the same as the model used
Nsim = int(8*60/ts) # set the simulation horizon
plant_model_file = './models/APPJmodel_TEOS_UCB_LAM_modord3.mat'
control_model_file = './models/APPJmodel_TEOS_UCB_LAM_modord3.mat'
# for testing new LTI model for experiments 
# (note: must change some parameters in my_system.py)
# plant_model_file = './models/2023_08_21_17h31m03s_APPJmodel.mat'
# control_model_file = './models/2023_08_21_17h31m03s_APPJmodel.mat'
#
filter_val = None#0.9#None

Fontsize = 14 # default font size for plots
Lwidth = 3 # default line width for plots

### SETUP: do not edit below this line, otherwise the reproducibility of the results is not guaranteed
## setup for establishing plotting defaults
lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

################################################################################
## NOMINAL MPC
################################################################################
# get problem information
prob_info = get_prob_info(
    plant_model_file=plant_model_file, 
    control_model_file=control_model_file,
    filter_val=filter_val,
    mpc_type='nominal',
    )

ts = prob_info['ts']
xss = prob_info['xss']
uss = prob_info['uss']
xssp = prob_info['xssp']
ussp = prob_info['ussp']
x_max = prob_info['x_max']
u_min = prob_info['u_min']
u_max = prob_info['u_max']

# get controller
c = NominalMPC(prob_info)
c.get_mpc()
c.set_parameters([np.zeros((3,)), np.zeros((2,1))])
res, feas = c.solve_mpc()
print(res)
print(feas)

# get observer
ekf = EKF(prob_info)
ekf.get_observer()

## run closed loop simulation using MPC
sim = Simulation(Nsim)
sim.load_prob_info(prob_info)
sim_data = sim.run_closed_loop(c, observer=ekf, offset=False, time_vary_filter=True)
print('Simulation Data Keys: ', [*sim_data])

ctime = sim_data['ctime']
print('Total Runtime: ', np.sum(ctime))
print('Average Runtime: ', np.mean(ctime))

Yrefplot = sim_data['Yrefsim']
Tref = Yrefplot[0,:] + xssp[0]
I706ref = Yrefplot[1,:] + xssp[1]
Tplot = sim_data['Ysim'][0,:] + xssp[0]
I706 = sim_data['Ysim'][1,:] + xssp[1]
I777 = sim_data['Ysim'][2,:] + xssp[2]
Pplot = sim_data['Usim'][0,:] + ussp[0]
qplot = sim_data['Usim'][1,:] + ussp[1]
fval = sim_data['fval_sim']

fig, axes = plt.subplots(3,2, sharex=True, figsize=(12,8))
axes[0,0].plot(np.arange(len(Tref))*ts, Tref, 'k--', label='Setpoint')
# axes[0,0].axhline(x_max[0]+xss[0], color='r', ls='--', label='Maximum')
axes[0,0].plot(np.arange(len(Tplot))*ts, Tplot, label='Nominal MPC')
axes[0,0].set_xlabel('Time (s)')
axes[0,0].set_ylabel('Surface Temperature\n('+r'$^\circ$'+'C)')

axes[0,1].plot(np.arange(len(I706ref))*ts, I706ref, 'k--', label='Setpoint')
axes[0,1].plot(np.arange(len(I706))*ts, I706, label='Nominal MPC')
axes[0,1].set_xlabel('Time (s)')
axes[0,1].set_ylabel('Intensity at He706\nPeak (arb. units)')

axes[1,0].plot(np.arange(len(I777))*ts, I777, label='Nominal')
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Intensity at O777\nPeak (arb. units)')

axes[1,1].plot(np.arange(len(fval))*ts, fval, color='g')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Filter Value')

axes[2,0].axhline(u_max[0]+uss[0], color='r', ls='--')
axes[2,0].text(0.9, 0.9, 'Maximum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,0].transAxes,
               color='r',
               )
axes[2,0].axhline(u_min[0]+uss[0], color='r', ls='--')
axes[2,0].text(0.9, 0.1, 'Minimum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,0].transAxes,
               color='r',
               )
axes[2,0].step(np.arange(len(Pplot))*ts, Pplot, label='Nominal')
axes[2,0].set_xlabel('Time (s)')
axes[2,0].set_ylabel('Applied Power (W)')

axes[2,1].axhline(u_max[1]+uss[1], color='r', ls='--')
axes[2,1].text(0.9, 0.9, 'Maximum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,1].transAxes,
               color='r',
               )
axes[2,1].axhline(u_min[1]+uss[1], color='r', ls='--')
axes[2,1].text(0.9, 0.1, 'Minimum', fontsize='small',
               horizontalalignment='center',
               verticalalignment='center', 
               transform=axes[2,1].transAxes,
               color='r',
               )
axes[2,1].step(np.arange(len(qplot))*ts, qplot, label='Nominal')
axes[2,1].set_xlabel('Time (s)')
axes[2,1].set_ylabel('Helium Flow Rate (SLM)')

plt.tight_layout()
plt.draw()
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(sim_data['Jsim'], label='Nominal')
plt.draw()


################################################################################
## OFFSET FREE MPC
################################################################################
# get problem information
prob_info = get_prob_info(
    plant_model_file=plant_model_file, 
    control_model_file=control_model_file,
    filter_val=filter_val,
    mpc_type='offsetfree',
    )

# get controller
c = OffsetFreeMPC(prob_info)
c.get_mpc()
c.set_parameters([np.zeros((3,)), np.zeros((2,1)), np.zeros((2,1))])
res, feas = c.solve_mpc()
print(res)
print(feas)

# get observer
ekf = EKF(prob_info)
ekf.get_observer()

## run closed loop simulation using MPC
sim = Simulation(Nsim)
sim.load_prob_info(prob_info)
sim_data = sim.run_closed_loop(c, observer=ekf, offset=True, time_vary_filter=True)
print('Simulation Data Keys: ', [*sim_data])

ctime = sim_data['ctime']
print('Total Runtime: ', np.sum(ctime))
print('Average Runtime: ', np.mean(ctime))

Yrefplot = sim_data['Yrefsim']
Tref = Yrefplot[0,:] + xssp[0]
I706ref = Yrefplot[1,:] + xssp[1]
Tplot = sim_data['Ysim'][0,:] + xssp[0]
I706 = sim_data['Ysim'][1,:] + xssp[1]
I777 = sim_data['Ysim'][2,:] + xssp[2]
Pplot = sim_data['Usim'][0,:] + ussp[0]
qplot = sim_data['Usim'][1,:] + ussp[1]
fval = sim_data['fval_sim']

axes[0,0].plot(np.arange(len(Tplot))*ts, Tplot, label='Offset-free MPC')
axes[0,0].legend(fontsize='small', loc='upper center')

axes[0,1].plot(np.arange(len(I706))*ts, I706, label='Offset-free MPC')
axes[0,1].legend(fontsize='small', loc='lower center')

axes[1,0].plot(np.arange(len(I777))*ts, I777, label='Offset-free')

axes[2,0].step(np.arange(len(Pplot))*ts, Pplot, label='Offset-free')

axes[2,1].step(np.arange(len(qplot))*ts, qplot, label='Offset-free')
# axes[2,1].legend(fontsize='small', loc='best')
plt.tight_layout()
plt.draw()
ax.plot(sim_data['Jsim'], label='Offset-free')
plt.draw()


plt.show()

# Import Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime
import time
from seabreeze.spectrometers import Spectrometer, list_devices
import serial
import asyncio

# import custom code
from config.my_system import get_prob_info_exp
from utils.controller import OffsetFreeMPC, NominalMPC
from utils.observer import EKF
import utils.APPJPythonFunctions as appj
from utils.experiments import Experiment


# defaults
STARTUP_DUTY_CYCLE = 100 # default duty cycle
STARTUP_POWER = 2.0 # default power
STARTUP_FLOW = 3.0 # default flow rate

### user inputs/options
ts = 1.0 # sampling time, ensure it is the same as the model used
Nsim = int(3*60/ts) # set the simulation horizon
mpc_type = 'offsetfree'
processing_info_file = './models/2023_08_21_17h31m03s_APPJ_model_train_data.mat'
control_model_file = './models/2023_08_21_17h31m03s_APPJmodel.mat'
collect_open_loop_data = False
run_test = True
Ts0_des = 37.0  # desired initial surface temperature to make consistent experiments
coolDownDiff = 1 # degrees to subtract from desired surface temperature for cooldown
warmUpDiff = 1 # degrees to subtract from desired surface temperature for warming up

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

if not collect_open_loop_data:
    # load the processing information used in model generation
    model = sio.loadmat(control_model_file, struct_as_record=False)
    dataInfo = model['dataInfo'][0][0]
    print(vars(dataInfo).keys())
    if 'InormFactor' in vars(dataInfo).keys():
        I_NORMALIZATION = 1/float(dataInfo.InormFactor)
    else:
        I_NORMALIZATION = 1.0
    print("OES normalization factor: ", I_NORMALIZATION)
    if 'baseline' in vars(dataInfo).keys():
        baseline = dataInfo.baseline
    else:
        baseline = 0.0

date = datetime.now().strftime('%Y_%m_%d_%H'+'h%M'+'m%S'+'s')
print(f'Timestamp for save files: {date}')
# set up Experimental Data folder
directory = os.getcwd()
saveDir = directory+"/ExperimentalData/"+date+"/"
print(f'\nData will be saved in the following directory:\n {saveDir}')

################################################################################
# APPJ STARTUP
################################################################################
# configure run options for gathering data
runOpts = appj.RunOpts()
runOpts.collectData = True
runOpts.collectEntireSpectra = True
runOpts.collectOscMeas = False
runOpts.collectSpatialTemp = False
runOpts.saveSpectra = True
runOpts.saveOscMeas = False
runOpts.saveSpatialTemp = False
runOpts.tSampling = ts

# connect to/open connection to devices in setup
# Arduino
arduinoAddress = appj.getArduinoAddress(os="ubuntu")
print("Arduino Address: ", arduinoAddress)
arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
print(arduinoPI)
s = time.time()
# # Oscilloscope
# oscilloscope = appj.Oscilloscope()       # Instantiate object from class
# instr = oscilloscope.initialize()	# Initialize oscilloscope
instr = None
# Spectrometer
devices = list_devices()
print(devices)
spec = Spectrometer(devices[0])
spec.integration_time_micros(12000*6)
# Thermal Camera
dev, ctx = appj.openThermalCamera()
print("Devices opened/connected to sucessfully!")

devices = {}
devices['arduinoPI'] = arduinoPI
devices['arduinoAddress'] = arduinoAddress
devices['instr'] = instr
devices['spec'] = spec

# send startup inputs
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
input("Ensure plasma has ignited and press Return to begin.\n")

## Startup asynchronous measurement
if os.name == 'nt':
    ioloop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
    asyncio.set_event_loop(ioloop)
else:
    ioloop = asyncio.get_event_loop()
# run once to initialize measurements
prevTime = (time.time()-s)*1e3
tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
print('measurement devices ready!')
s = time.time()

# let APPJ run for a bit
STARTUP_POWER = 3.0
STARTUP_FLOW = 3.0
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
time.sleep(0.5)
appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)

w8 = input("Wait 5 min? [y,n]\n")
if w8 == 'y':
    print("Waiting 5 minutes to ensure roughly steady plasma startup...\n")
    time.sleep(60)
    print("4 minutes left...")
    time.sleep(60)
    print("3 minutes left...")
    time.sleep(60)
    print("2 minutes left...")
    time.sleep(60)
    print("1 minute left...")
    time.sleep(60)
else:
    time.sleep(5)

# # wait for cooldown
# appj.sendInputsArduino(arduinoPI, 0.0, 0.0, STARTUP_DUTY_CYCLE, arduinoAddress)
# arduinoPI.close()
# while appj.getSurfaceTemperature() > Ts0_des-coolDownDiff:
    # time.sleep(runOpts.tSampling)
    # print('cooling down ...')
# arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
# time.sleep(2)
# appj.sendInputsArduino(arduinoPI, STARTUP_POWER, STARTUP_FLOW, STARTUP_DUTY_CYCLE, arduinoAddress)
# # wait for surface to reach desired starting temp
# while appj.getSurfaceTemperature() < Ts0_des-warmUpDiff:
    # time.sleep(runOpts.tSampling)
    # print('warming up ...')

prevTime = (time.time()-s)*1e3
# get initial measurements
tasks, runTime = ioloop.run_until_complete(appj.async_measure(arduinoPI, prevTime, instr, spec, runOpts))
if runOpts.collectData:
    thermalCamOut = tasks[0].result()
    Ts0 = thermalCamOut[0]
    specOut = tasks[1].result()
    if collect_open_loop_data:
        I0 = specOut[0]
    else:
        I0 = specOut[0]*I_NORMALIZATION
    oscOut = tasks[2].result()
    arduinoOut = tasks[3].result()
    outString = "Measured Outputs: Temperature: %.2f, Intensity: %.2f" % (Ts0, I0)
    print(outString)
else:
    Ts0 = 37
    I0 = 100

s = time.time()

# arduinoPI.close()

################################################################################
# PROBLEM SETUP
################################################################################
if not collect_open_loop_data:
    # get problem information
    prob_info = get_prob_info_exp(
        processing_info_file=processing_info_file,
        control_model_file=control_model_file,
        mpc_type = mpc_type,
        )

    # get controller
    if mpc_type == 'nominal':
        c = NominalMPC(prob_info)
    elif mpc_type == 'offsetfree':
        c = OffsetFreeMPC(prob_info)
    c.get_mpc()
    c.set_parameters([np.zeros((3,)), np.zeros((2,1)), np.zeros((2,1))])
    res, feas = c.solve_mpc()
    print(res)
    print(feas)

    # get observer
    ekf = EKF(prob_info)
    ekf.get_observer()

if any([collect_open_loop_data, run_test]):
    ############################################################################
    ## Begin Experiment: Experiment with generated hardware code
    ############################################################################
    exp = Experiment(Nsim, saveDir)

    # arduinoPI = serial.Serial(arduinoAddress, baudrate=38400, timeout=1)
    # devices['arduinoPI'] = arduinoPI

    if collect_open_loop_data:
        # create input sequences
        uvec1 = np.linspace(1.5,3.5,5) # for power
        uvec2 = np.linspace(1.5,5.5,9) # for flow rate
        uu1,uu2 = np.meshgrid(uvec1,uvec2)
        uvec1 = uu1.reshape(-1,)
        uvec2 = uu2.reshape(-1,)
        rng = np.random.default_rng(0)
        rng.shuffle(uvec1)
        pseq = np.copy(uvec1)
        pseq = np.insert(pseq,0,[0.0,2.5,2.5,2.5])
        rng.shuffle(uvec2)
        qseq = np.copy(uvec2)
        qseq = np.insert(qseq,0,[0.0,3.5,3.5,3.5])
        print(pseq)
        print(qseq)

        pseq = np.repeat(pseq, 45/runOpts.tSampling).reshape(-1,)
        qseq = np.repeat(qseq, 45/runOpts.tSampling).reshape(-1,)
        print(pseq.shape[0])

        prevTime = (time.time()-s)*1e3
        exp_data = exp.run_open_loop(ioloop,
                                        power_seq=pseq,
                                        flow_seq=qseq,
                                        runOpts=runOpts,
                                        devices=devices,
                                        prevTime=prevTime,
                                        )
    else:
        exp.load_prob_info(prob_info)
        prevTime = (time.time()-s)*1e3
        exp_data = exp.run_closed_loop_mpc(c, 
                                           ioloop,
                                           observer=ekf,
                                           runOpts=runOpts,
                                           devices=devices,
                                           prevTime=prevTime,
                                           I_NORM=I_NORMALIZATION,
                                           )

# turn off plasma jet (programmatically)
appj.sendInputsArduino(arduinoPI, 0.0, 0.0, STARTUP_DUTY_CYCLE, arduinoAddress)
appj.sendInputsArduino(arduinoPI, 0.0, 0.0, STARTUP_DUTY_CYCLE, arduinoAddress)
arduinoPI.close()
print(f"Experiments complete at {datetime.now().strftime('%Y_%m_%d_%H'+'h%M'+'m%S'+'s')}!\n"+
    "################################################################################################################\n"+
    "IF FINISHED WITH EXPERIMENTS, PLEASE FOLLOW THE SHUT-OFF PROCEDURE FOR THE APPJ\n"+
    "################################################################################################################\n")

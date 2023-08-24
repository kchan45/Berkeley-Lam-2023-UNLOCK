# problem details
#


#altering y_max to be done in this section of the code!!! - 
#has to be updated in dummy_sys.py every-time and not in the main FilmDepControl ipynb code

#y_max = np.array([80,2500,2500]) - xss
#y_max = np.array([60,50000,50000]) - xss
#altering y_max    


#
# Requirements:
# * Python 3
# * CasADi [https://web.casadi.org]
#
# Copyright (c) 2022 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import numpy as np
import scipy.linalg
from scipy import io
import casadi as cas

from config.reference_signal import myRef

def get_prob_info(ref=1.0, alpha=1.0, colab=True):

    ts = 1 # sampling time (in seconds)
    rand_seed = 520

    Np = 5      # Prediction horizon

    ## load system matrices from Data model ID
    if colab:
        
        #In the first modification I made
        #modelp = io.loadmat('./APPJ/Valid_split_005.mat')
        #model = io.loadmat('./APPJ/Valid_split_025.mat')  
        #In the first modification I made
        
        #In the zeroth modification I made - but hardly any difference
        modelp = io.loadmat('./APPJ/APPJmodel_TEOS_UCB_LAM_modord3.mat')
        model = io.loadmat('./APPJ/APPJmodel_TEOS_UCB_LAM_modord3_half.mat')  

        #In the zeroth modification I made - but hardly any difference

        #In original code sent by Kimberly
        #modelp = io.loadmat('/content/drive/MyDrive/Research/Berkeley-Lam/APPJ/APPJmodel_TEOS_UCB_LAM_modord3.mat')
        #model = io.loadmat('/content/drive/MyDrive/Research/Berkeley-Lam/APPJ/APPJmodel_TEOS_UCB_LAM_modord3_half.mat')
        #In original code sent by Kimberly


    else:
        modelp = io.loadmat('./APPJ/APPJmodel_TEOS_UCB_LAM_modord3.mat')
        model = io.loadmat('./APPJ/APPJmodel_TEOS_UCB_LAM_modord3.mat')

    A = model['A']
    B = model['B']
    C = model['C']
    xss = np.ravel(model['yss']) # [Ts; I(706); I(777)]
    uss = np.ravel(model['uss']) # [P; q]
    print('Linear Model to be used for CONTROL:')
    print('A: ', A)
    print('B: ', B)
    print('C: ', C)
    print('xss: ', xss)
    print()

    Ap = modelp['A']
    Bp = modelp['B']
    Cp = modelp['C']
    xssp = np.ravel(modelp['yss']) # [Ts; I(706); I(777)]
    ussp = np.ravel(modelp['uss']) # [P; q]
    print('Linear Model to be used for the PLANT:')
    print('A: ', Ap)
    print('B: ', Bp)
    print('C: ', Cp)
    print('xss: ', xssp)

    #print('Cp')
    #print(type(Cp))
    
    
    myref = lambda t: myRef(t, ts, ref=ref) # reference signal

    nx = A.shape[1] # number of states
    nu = B.shape[1] # number of inputs (q, P)
    ny = C.shape[0] # number of outputs (Ts, I(706), I(777))
    nyc = 1         # number of controlled outputs
    nd = 0          # offset-free disturbances
    nw = nx         # process noise
    nv = ny         # measurement noise

#    x0 = np.zeros((nx,)) # initial state
    x0 = np.array([0, -100, -300])
    #np.zeros((nx,)) # initial state

    ## load/set MPC info
    # constraint bounds
    u_min = np.array([1.5, 1.5]) - uss
    u_max = np.array([5,5]) - uss
    y_min = np.array([25,-100,-100]) - xss

    #altering y_max
    #y_max = np.array([80,2500,2500]) - xss
    y_max = np.array([60,50000,4500]) - xss
    #altering y_max    
    
    x_min = y_min#-np.inf*np.ones((nx,))
    x_max = y_max#np.inf*np.ones((nx,))
    # v_min = 0*-0.01*np.ones(nv)
    # v_max = 0*0.01*np.ones(nv)
    
    #
    #
    #
    #
    #
    v_mu = 0
    v_sigma = 0
    w_min = 0*np.ones(nw)
    w_max = 0*np.ones(nw)
    
    #v_mu = 0
    #v_sigma = 0.1
    #w_min = -1*np.ones(nw)
    #w_max = 1*np.ones(nw)
    
    
    

    # initial variable guesses
    u_init = (u_min+u_max)/2
    
    x_init = np.zeros((nx,))#(x_min+x_max)/2
    y_init = (y_min+y_max)/2
    

    ## create casadi functions for problem
    # casadi symbols
    x = cas.SX.sym('x', nx)
    u = cas.SX.sym('u', nu)
    w = cas.SX.sym('w', nw)
    wp = cas.SX.sym('wp', nw) # predicted uncertainty
    v = cas.SX.sym('v', nv)
    yref = cas.SX.sym('yref', nyc)
    

    # dynamics function (prediction model)
    xnext = A@x + B@u + wp
    f = cas.Function('f', [x,u,wp], [xnext])

    # output equation (for control model)
    y = C@x
    h = cas.Function('h', [x], [y])

    # controlled output equation
    ymeas = cas.SX.sym('ymeas', ny)
    yc = ymeas[0]
    r = cas.Function('r', [ymeas], [yc])

    # plant model
    xnextp = Ap@x + Bp@u + w
    fp = cas.Function('fp', [x,u,w], [xnextp])

    #
    #
    i = cas.SX.sym('i', 1)
    n = cas.SX.sym('n', 1)
    #
    
    # output equation (for plant)    
    yp = ((n-i)/n)*(Cp@x) + v
    
    #yp = (i/n)*(Cp@x) + v
    hp = cas.Function('hp', [x,v,i,n], [yp])
    
    #yp = Cp@x + v
    #hp = cas.Function('hp', [x,v], [yp])
    
    # dep rate output
    #dh = alpha*((x[1]+xssp[1])/(x[2]+xssp[2]))*ts/60
    #dh = x[1]+xssp[1]
    dh = 0
    # dh = alpha*(x[1]+xssp[1]+w[1])*ts/60
    deltaH = cas.Function('deltaH', [x], [dh])

    # stage cost (nonlinear thickness change computation)
    
    lstg = x[1]+xss[1]+w[1]
    # lstg = alpha*(x[1]+xss[1]+w[1])*ts/60 # using He(706) peak
    lstage = cas.Function('lstage', [x,w], [lstg])

        
    lstg2 = x[0]+xss[0]+w[0]
    # lstg = alpha*(x[1]+xss[1]+w[1])*ts/60 # using He(706) peak
    lstage2 = cas.Function('lstage2', [x,w], [lstg2])

    
    warm_start = True

    ## pack away problem info
    prob_info = {}
    prob_info['Np'] = Np
    prob_info['myref'] = myref

    prob_info['ts'] = ts
    prob_info['x0'] = x0
    prob_info['rand_seed'] = rand_seed

    prob_info['nu'] = nu
    prob_info['nx'] = nx
    prob_info['ny'] = ny
    prob_info['nyc'] = nyc
    prob_info['nv'] = nv
    prob_info['nw'] = nw
    prob_info['nd'] = nd

    prob_info['u_min'] = u_min
    prob_info['u_max'] = u_max
    prob_info['x_min'] = x_min
    prob_info['x_max'] = x_max
    prob_info['y_min'] = y_min
    prob_info['y_max'] = y_max
    # prob_info['v_min'] = v_min
    # prob_info['v_max'] = v_max
    prob_info['v_mu'] = v_mu
    prob_info['v_sigma'] = v_sigma
    prob_info['w_min'] = w_min
    prob_info['w_max'] = w_max

    prob_info['u_init'] = u_init
    prob_info['x_init'] = x_init
    prob_info['y_init'] = y_init

    prob_info['f'] = f
    prob_info['h'] = h
    prob_info['r'] = r
    prob_info['fp'] = fp
    prob_info['hp'] = hp
    prob_info['deltaH'] = deltaH
    prob_info['lstage'] = lstage
    prob_info['lstage2'] = lstage2
    prob_info['warm_start'] = warm_start

    prob_info['xssp'] = xssp
    prob_info['ussp'] = ussp
    prob_info['xss'] = xss
    prob_info['uss'] = uss
    
    return prob_info

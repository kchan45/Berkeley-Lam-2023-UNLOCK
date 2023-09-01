# problem details
#
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
from scipy import io
import casadi as cas

from config.reference_signal import myRef

def get_prob_info(
        control_model_file='', 
        plant_model_file='', 
        filter_val=None, 
        mpc_type='offsetfree',
    ):

    ts = 1 # sampling time (in seconds)
    rand_seed = 520

    Np = 10      # Prediction horizon

    ## load system matrices from Data model ID
    modelp = io.loadmat(plant_model_file)
    model = io.loadmat(control_model_file)

    A = model['A']
    B = model['B']
    C = model['C'].astype(np.float64)
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
    Cp = modelp['C'].astype(np.float64)
    if filter_val is not None:
        Cp[1,1] = filter_val
        Cp[2,2] = filter_val
    xssp = np.ravel(modelp['yss']) # [Ts; I(706); I(777)]
    ussp = np.ravel(modelp['uss']) # [P; q]
    print('Linear Model to be used for the PLANT:')
    print('A: ', Ap)
    print('B: ', Bp)
    print('C: ', Cp)
    print('xss: ', xssp)

    nx = A.shape[1] # number of states
    nu = B.shape[1] # number of inputs (q, P)
    ny = C.shape[0] # number of outputs (Ts, I(706), I(777))
    nyc = 2         # number of controlled outputs
    if mpc_type == 'offsetfree':
        nd = nyc          # offset-free disturbances
    else:
        nd = 0
    nw = nx         # process noise
    nv = ny         # measurement noise

    myref = lambda t: myRef(t, ts, ref=np.array([55.0,3000.0])) - xss[:nyc] # reference signal
    # myref = lambda t: myRef(t, ts, ref=xss[:nyc]) - xss[:nyc] # reference signal
    
    # ------ use this for new LTI model for experiments -------
    # myref = lambda t: myRef(t, ts, ref=np.array([0.6, -0.2])) - xss[:nyc] # reference signal
    # ------ use this for new LTI model for experiments -------

    x0 = np.array([58.0,2950.0,4400.0]) - xss#np.zeros((nx,)) # initial state

    ## load/set MPC info
    # constraint bounds
    u_min = np.array([1.5, 1.5]) - uss
    u_max = np.array([5,5]) - uss
    du_min = np.array([-0.5, -0.5])
    du_max = np.array([0.5,0.5])
    y_min = np.array([25,0.0,0.0]) - xss
    y_max = np.array([65,5000,5000]) - xss
    x_min = y_min#-np.inf*np.ones((nx,))
    x_max = y_max#np.inf*np.ones((nx,))
    # v_min = 0*-0.01*np.ones(nv)
    # v_max = 0*0.01*np.ones(nv)
    v_mu = 0
    v_sigma = 0.1
    w_min = 0.0*np.ones((nw,))
    w_max = 0.0*np.ones((nw,))

    # ------ use this for new LTI model for experiments -------
    # y_min = -1*np.ones((ny,))-xss
    # y_max = np.ones((ny,))-xss
    # v_mu = 0
    # v_sigma = 0.01
    # ------ use this for new LTI model for experiments -------

    # initial variable guesses
    u_init = (u_min+u_max)/2
    x_init = np.zeros((nx,))#(x_min+x_max)/2
    y_init = (y_min+y_max)/2#np.array([30, 1000, 1000])#

    ## create casadi functions for problem
    # casadi symbols
    x = cas.SX.sym('x', nx)
    u = cas.SX.sym('u', nu)
    d = cas.SX.sym('d', nd)
    w = cas.SX.sym('w', nw)
    v = cas.SX.sym('v', nv)
    x_ss = cas.SX.sym('x_ss', nx)
    u_ss = cas.SX.sym('u_ss', nu)
    fval = cas.SX.sym('fval', 1)
    yref = cas.SX.sym('yref', nyc)

    # dynamics function (prediction model)
    xnext = A@x + B@u
    if mpc_type == 'offsetfree':
        if nd > 0:
            xnext[:nd] = xnext[:nd] + d
    f = cas.Function('f', [x,u,d], [xnext])

    # output equation (for control model)
    y = C@x
    h = cas.Function('h', [x,d], [y])


    # controlled output equation
    ymeas = cas.SX.sym('ymeas', ny)
    yc = ymeas[:nyc]
    r = cas.Function('r', [ymeas], [yc])

    # plant model
    xnextp = Ap@x + Bp@u + w
    fp = cas.Function('fp', [x,u,w], [xnextp])

    # output equation (for plant)
    yp = x + v
    yp[0] = x[0] + v[0]
    yp[1] = fval*x[1] + v[1]
    yp[2] = fval*x[2] + v[2]
    hp = cas.Function('hp', [x,v,fval], [yp])

    if mpc_type == 'offsetfree':
        # stage cost (reference tracking)
        Q = 1.0*np.eye(nx)
        Q[1,1] = 1e-2*Q[1,1]
        Q[-1,-1] = 0.0
        R = 1.0*np.eye(nu)
        lstg = (x-x_ss).T @ Q @ (x-x_ss) + (u-u_ss).T @ R @ (u-u_ss)
        lstage = cas.Function('lstage', [x,u,x_ss,u_ss], [lstg])

        # terminal cost
        P = 0*np.eye(nx)
        ltrm = (x-x_ss).T @ P @ (x-x_ss)
        lterm = cas.Function('lterm', [x,x_ss], [ltrm])

    elif mpc_type == 'nominal':
        lstg = cas.sumsqr(r(y)-yref)
        lstage = cas.Function('lstage', [x,yref], [lstg])

        ltrm = cas.sumsqr(r(y)-yref)
        lterm = cas.Function('lterm', [x,yref], [ltrm])

    term_eq_cons = True
    target_penalty = 1.0e1
    warm_start = False

    # observer
    Qobs = 1e-7 * np.eye(nx+nd)
    Robs = 1e-6 * np.eye(ny)

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
    # prob_info['du_min'] = du_min
    # prob_info['du_max'] = du_max
    prob_info['x_min'] = x_min
    prob_info['x_max'] = x_max
    prob_info['y_min'] = y_min
    prob_info['y_max'] = y_max
    prob_info['yc_min'] = y_min[:nyc]
    prob_info['yc_max'] = y_max[:nyc]
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
    prob_info['filter_val'] = filter_val or 1.0
    prob_info['stage_cost'] = lstage
    prob_info['term_cost'] = lterm
    prob_info['term_eq_cons'] = term_eq_cons
    prob_info['target_penalty'] = target_penalty
    prob_info['warm_start'] = warm_start

    prob_info['Qobs'] = Qobs 
    prob_info['Robs'] = Robs

    prob_info['xssp'] = xssp
    prob_info['ussp'] = ussp
    prob_info['xss'] = xss
    prob_info['uss'] = uss

    return prob_info

def get_prob_info_exp(
        control_model_file='', 
        processing_info_file='', 
        mpc_type='offsetfree',
    ):

    ts = 1 # sampling time (in seconds)
    rand_seed = 520

    Np = 10      # Prediction horizon

    ## load system matrices from Data model ID
    model = io.loadmat(control_model_file)

    A = model['A']
    B = model['B']
    C = model['C'].astype(np.float64)
    xss = np.ravel(model['yss']) # [Ts; I(706); I(777)]
    uss = np.ravel(model['uss']) # [P; q]
    print('Linear Model to be used for CONTROL:')
    print('A: ', A)
    print('B: ', B)
    print('C: ', C)
    print('xss: ', xss)
    print()

    nx = A.shape[1] # number of states
    nu = B.shape[1] # number of inputs (q, P)
    ny = C.shape[0] # number of outputs (Ts, I(706), I(777))
    nyc = 2         # number of controlled outputs
    if mpc_type == 'offsetfree':
        nd = nyc          # offset-free disturbances
    else:
        nd = 0
    nw = nx         # process noise
    nv = ny         # measurement noise
    
    ref_vals = np.array([55.0,1000.0])
    
    ## output processing from measurements
    ymeas = cas.SX.sym('ymeas', ny)
    processing_info = io.loadmat(processing_info_file)
    mean_shift = np.zeros((ny,))
    if 'mean_shift' in processing_info.keys():
        mean_shift = processing_info['mean_shift']
    background = np.zeros((ny,1))
    if 'background' in processing_info.keys():
        background = np.ravel(processing_info['background'])
        y2_idx = processing_info['I706idx']
        y3_idx = processing_info['I777idx']
    
    y2 = ymeas[1] - mean_shift - background[y2_idx]
    y3 = ymeas[2] - mean_shift - background[y3_idx]
    yc = cas.vertcat(ymeas[0], y2, y3)
    
    ref_vals[1] = ref_vals[1] - mean_shift - background[y2_idx]
    
    if 'y_min' in processing_info.keys():
        y_min_scale = processing_info['y_min']
        y_max_scale = processing_info['y_max']
        yc_proc = 2*(yc-y_min_scale)/(y_max_scale-y_min_scale) - 1
        y_min_scale = np.ravel(y_min_scale)
        y_max_scale = np.ravel(y_max_scale)
        ref_vals = 2*(ref_vals[:nyc]-y_min_scale[:nyc])/(y_max_scale[:nyc]-y_min_scale[:nyc]) - 1
    else:
        yc_proc = yc

    output_proc = cas.Function('output_proc', [ymeas], [yc_proc])
    
    print(ref_vals)
    ref_vals = np.array([0.3, -0.1])
    myref = lambda t: myRef(t, ts, ref=ref_vals) - xss[:nyc] # reference signal
    # myref = lambda t: myRef(t, ts, ref=xss[:nyc]) - xss[:nyc] # reference signal

    x0 = np.zeros((nx,)) # initial state

    ## load/set MPC info
    # constraint bounds
    u_min = np.array([1.5, 1.5]) - uss
    u_max = np.array([3.5, 5.5]) - uss
    du_min = np.array([-0.5, -0.5])
    du_max = np.array([0.5,0.5])
    y_min = -1.0*np.ones((ny,)) - xss #np.array([25,0.0,0.0]) - xss
    y_max = 1.0*np.ones((ny,)) - xss #np.array([65,5000,5000]) - xss
    x_min = y_min#-np.inf*np.ones((nx,))
    x_max = y_max#np.inf*np.ones((nx,))

    # initial variable guesses
    u_init = (u_min+u_max)/2
    x_init = np.zeros((nx,))#(x_min+x_max)/2
    y_init = (y_min+y_max)/2

    ## create casadi functions for problem
    # casadi symbols
    x = cas.SX.sym('x', nx)
    u = cas.SX.sym('u', nu)
    d = cas.SX.sym('d', nd)
    x_ss = cas.SX.sym('x_ss', nx)
    u_ss = cas.SX.sym('u_ss', nu)
    yref = cas.SX.sym('yref', nyc)

    # dynamics function (prediction model)
    xnext = A@x + B@u
    if mpc_type == 'offsetfree':
        if nd > 0:
            xnext[:nd] = xnext[:nd] + d
    f = cas.Function('f', [x,u,d], [xnext])

    # output equation (for control model)
    y = C@x
    h = cas.Function('h', [x,d], [y])

    # controlled output equation
    yc = ymeas[:nyc]
    r = cas.Function('r', [ymeas], [yc])

    if mpc_type == 'offsetfree':
        # stage cost (reference tracking)
        Q = 1.0*np.eye(nx)
        Q[1,1] = 1e-2*Q[1,1]
        Q[-1,-1] = 0.0
        R = 1.0*np.eye(nu)
        lstg = (x-x_ss).T @ Q @ (x-x_ss) + (u-u_ss).T @ R @ (u-u_ss)
        lstage = cas.Function('lstage', [x,u,x_ss,u_ss], [lstg])

        # terminal cost
        P = 0*np.eye(nx)
        ltrm = (x-x_ss).T @ P @ (x-x_ss)
        lterm = cas.Function('lterm', [x,x_ss], [ltrm])

    elif mpc_type == 'nominal':
        lstg = cas.sumsqr(r(y)-yref)
        lstage = cas.Function('lstage', [x,yref], [lstg])

        ltrm = cas.sumsqr(r(y)-yref)
        lterm = cas.Function('lterm', [x,yref], [ltrm])

    term_eq_cons = True
    target_penalty = 1.0e1
    warm_start = False

    ## observer
    Qobs = 1e-7 * np.eye(nx+nd)
    Robs = 1e-6 * np.eye(ny)

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
    # prob_info['du_min'] = du_min
    # prob_info['du_max'] = du_max
    prob_info['x_min'] = x_min
    prob_info['x_max'] = x_max
    prob_info['y_min'] = y_min
    prob_info['y_max'] = y_max
    prob_info['yc_min'] = y_min[:nyc]
    prob_info['yc_max'] = y_max[:nyc]

    prob_info['u_init'] = u_init
    prob_info['x_init'] = x_init
    prob_info['y_init'] = y_init

    prob_info['f'] = f
    prob_info['h'] = h
    prob_info['r'] = r
    prob_info['output_proc'] = output_proc
    prob_info['I706idx'] = y2_idx
    prob_info['I777idx'] = y3_idx
    prob_info['stage_cost'] = lstage
    prob_info['term_cost'] = lterm
    prob_info['term_eq_cons'] = term_eq_cons
    prob_info['target_penalty'] = target_penalty
    prob_info['warm_start'] = warm_start

    prob_info['Qobs'] = Qobs 
    prob_info['Robs'] = Robs

    prob_info['xss'] = xss
    prob_info['uss'] = uss

    return prob_info

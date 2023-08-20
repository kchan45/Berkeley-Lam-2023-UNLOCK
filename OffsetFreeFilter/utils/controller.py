# model predictive controller
#
# This script provides definitions of classes that can be used for model
# predictive control (MPC) schemes. This script is part of the package of code
# that produces the results of the following paper:
#
#
# Requirements:
# * Python 3
# * CasADi [https://web.casadi.org]
#
# Copyright (c) 2021 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import sys
sys.dont_write_bytecode = True
import os
import casadi as cas
import numpy as np
import itertools


class MPC():
    """
    MPC is a super class designed to be a template for particular
    implementations of model predictive controllers (MPCs). Users should develop
    their own MPCs by using the general structure/methods provided below. Upon
    initialization of this class or any of its child classes, users should
    provide a Python dict that contains all of the relevant problem information.
    This class is designed to be used with CasADi to generate the optimization
    problems associated with MPC. Users are referred to the CasADi documentation
    for more information on CasADi.
    """

    def __init__(self, prob_info):
        super(MPC, self).__init__()
        self.prob_info = prob_info
        self.mpc_type = None

    def get_mpc(self, arg):
        """
        This method should generate the MPC problem by unpacking relevant
        information from the prob_info dict defined upon instantiation of the
        class. Upon defining the optimization problem, this method should save
        and/or return the appropriate objects such that this object may be
        called upon later. (e.g. if using the Opti stack interface of CasADi,
        save/return the objects that reference the optimization object
        (typically named opti) and the optimization variable references)
        """
        pass

    def reset_initial_guesses(self, arg):
        """
        This method should reset any initial guesses of the decision variables
        passed into the optimization problem. This method provides a way to
        simulate repeated solves of the optimization problem without re-defining
        an entirely new problem. If using the Opti stack interface of CasADi,
        this method mainly involves using the set_initial() method of the Opti
        object.
        """
        pass

    def set_parameters(self, arg):
        """
        This method should (re)set any parameters in the optimization problem.
        This method provides a way to simulated consistent and repeated solves
        of the optimization problem without re-defining an entirely new problem.
        If using the Opti stack interface of CasADi, this method mainly involves
        using the set_value() method of the Opti object.
        """
        pass

    def solve_mpc(self, arg):
        """
        This method should solve the optimization problem and return/save the
        relavent optimal variables. For MPC, this is typically the first optimal
        input determined by the solver. Users may also wish to return other
        values of the optimization problem and/or the entire solution of the
        optimization problem. This method should also handle any Exceptions that
        may occur upon a call to solve the optimization problem in the form of a
        try/except clause. If using the Opti stack interface of CasADi, this
        method mainly involves the call to the solve() method of the Opti
        object, as well as calls to the value() method of OptiSolution/Opti
        objects.
        """
        pass

class NominalMPC(MPC):
    def __init__(self, prob_info):
        super(NominalMPC, self).__init__(prob_info)
        self.mpc_type = 'nominal'

    def get_mpc(self):
        """
        """
        ## unpack relevant problem information
        # prediction horizon
        Np = self.prob_info['Np']
        # system dimensions
        nu = self.prob_info['nu']
        nx = self.prob_info['nx']
        ny = self.prob_info['ny']
        nyc = self.prob_info['nyc']
        nd = self.prob_info['nd']
        # state/input/output constraints
        u_min = self.prob_info['u_min']
        u_max = self.prob_info['u_max']
        x_min = self.prob_info['x_min']
        x_max = self.prob_info['x_max']
        y_min = self.prob_info['y_min']
        y_max = self.prob_info['y_max']
        # initial guesses for solver
        u_init = self.prob_info['u_init']
        x_init = self.prob_info['x_init']
        y_init = self.prob_info['y_init']
        # dynamic equations (state evolution and output equation)
        f = self.prob_info['f']
        h = self.prob_info['h']
        r = self.prob_info['r']
        # OCP costs
        stage_cost = self.prob_info['stage_cost']
        term_cost = self.prob_info['term_cost']
        # additional input cost/objectives and/or constraints
        reduce_dinput = False
        constrain_dinput = False
        if 'ustage' in self.prob_info.keys():
            ustage = self.prob_info['ustage']
            reduce_dinput = True
        if 'du_max' in self.prob_info.keys():
            du_max = self.prob_info['du_max']
            du_min = self.prob_info['du_min']
            constrain_dinput = True

        # create NLP opti object
        opti = cas.Opti()

        # Initialize container lists for all states, inputs, and outputs
        X = [0 for j in range(Np+1)]
        # Y = [0 for j in range(Np+1)]
        U = [0 for j in range(Np)]

        J = 0 # initialize cost/objective function

        ## define parameter(s), variable(s), and problem
        X[0] = opti.parameter(nx) # initial state as a parameter
        opti.set_value(X[0], np.zeros((nx,1)))

        # Y[0] = opti.variable(ny) # initial output variable
        # opti.subject_to(Y[0] == h(X[0]))
        # opti.set_initial(Y[0], y_init)

        Yref = opti.parameter(nyc)
        opti.set_value(Yref, np.zeros((nyc,1)))

        # the loop below systematically defines the variables of the optimal
        # control problem (OCP) over the prediction horizon
        for k in range(Np):
            # variable and constraints for u_{k}
            U[k] = opti.variable(nu)
            opti.subject_to(opti.bounded(u_min, U[k], u_max))
            opti.set_initial(U[k], u_init)

            Jstage = stage_cost(X[k], Yref)
            J += Jstage # add to the cost (construction of the objective)

            # variable x_{k+1}
            X[k+1] = opti.variable(nx)
            opti.subject_to(opti.bounded(x_min, X[k+1], x_max))
            opti.set_initial(X[k+1], x_init)

            # # variable y_{k+1}
            # Y[k+1] = opti.variable(ny)
            # opti.subject_to(opti.bounded(y_min, Y[k+1], y_max))
            # opti.set_initial(Y[k+1], y_init)

            # dynamics constraint
            opti.subject_to(X[k+1] == f(X[k],U[k], 0.0))

            # # output equality constraint
            # opti.subject_to(Y[k+1] == h(X[k+1],D))

            if k>0 and reduce_dinput:
                J += ustage(U[k],U[k-1])
            elif k>0 and constrain_dinput:
                opti.subject_to(opti.bounded(du_min, U[k]-U[k-1], du_max))

        # terminal cost and constraints
        J_end = term_cost(X[-1], Yref)
        J += J_end

        # set to minimize objective/cost
        opti.minimize( J )

        # set solver options
        p_opts = {'verbose': 0,
                  'expand': True,
                  'print_time': 0}          # problem options
        s_opts = {'max_iter': 1000,
                  'print_level': 0,
                  'tol': 1e-6}              # solver options
        opti.solver('ipopt', p_opts, s_opts) # add the solver to the opti object

        # save list containers of variables/parameters into a dict for portability
        opti_vars = {}
        opti_vars['U'] = U
        opti_vars['X'] = X[1:]
        # opti_vars['Y'] = Y
        opti_vars['J'] = J

        opti_params  = {}
        opti_params['X0'] = X[0]
        opti_params['Yref'] = Yref

        # save opti object and variable containers as attributes of NominalMPC
        # object
        self.opti = opti
        self.opti_vars = opti_vars
        self.opti_params = opti_params

        return opti, opti_vars, opti_params
    
    def reset_initial_guesses(self):
        """
        This method resets the initial guesses for the variabls in the optimization problem back to those defined in the prob_info dictionary provided upon instantiation of the OffsetFreeMPC object
        """
        # unpack relevant information from the prob_info dictionary
        Np = self.prob_info['Np']
        u_init = self.prob_info['u_init']
        x_init = self.prob_info['x_init']
        # y_init = self.prob_info['y_init']

        # unpack relevant variable containers from problem creation
        U = self.opti_vars['U']
        X = self.opti_vars['X']
        # Y = self.opti_vars['Y']
        
        for k in range(Np):
            self.opti.set_initial(U[k], u_init)
            self.opti.set_initial(X[k], x_init)
        #     self.opti.set_initial(Y[k], y_init)
        # self.opti.set_initial(Y[k+1], y_init)

    def set_parameters(self, params_list):
        """
        This method sets the values of the parameters of the optimization
        problem to those provided as arguments to this method. The argument
        params_list is a list of new parameter values to set in the same order
        as the opti_param keys.
        """

        # unpack parameter containers
        X0 = self.opti_params['X0']
        Yref = self.opti_params['Yref']

        # unpack params_list argument
        x0 = params_list[0]
        yref = params_list[1]

        self.opti.set_value(Yref, yref)

        # reset initial condition
        self.opti.set_value(X0, x0)

    def solve_mpc(self):
        """
        This method solves the MPC as defined by the get_mpc() method of this
        class. This method can only be called after the the get_mpc() method has
        been called (i.e., the optimization problme must be defined before it
        can be solved).
        """
        # extract all keys from the opti variables dict
        opti_var_keys = [*self.opti_vars]
        opti_param_keys = [*self.opti_params]

        # unpack relevant information from problem creation
        u_min = self.prob_info['u_min']
        u_max = self.prob_info['u_max']

        feas = True
        res = {}
        try:
            sol = self.opti.solve()

            for key in opti_var_keys:
                if key in ['J', 'Xss', 'Uss', 'Yss', 'Ycss']:
                    res[key] = sol.value(self.opti_vars[key])
                else:
                    var = self.opti_vars[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j,] = sol.value(var[j])

                    res[key] = values

            res['Ufull'] = res['U']
            res['U'] = res['U'][:,0]

            for key in opti_param_keys:
                if key in ['X0', 'Yref', 'D']:
                    res[key] = sol.value(self.opti_params[key])
                else:
                    var = self.opti_params[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j] = sol.value(var[j])

                    res[key] = values

            if self.prob_info['warm_start']:
                self.opti.set_initial(sol.value_variables())

        except Exception as e:
            print(e)
            # if solve fails, get the last value
            feas = False

            for key in opti_var_keys:
                if key in ['J', 'Xss', 'Uss', 'Yss', 'Ycss']:
                    res[key] = self.opti.debug.value(self.opti_vars[key])
                else:
                    var = self.opti_vars[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j] = self.opti.debug.value(var[j])

                    res[key] = values

            res['Ufull'] = res['U']
            res['U'] = res['U'][:,0]

            for key in opti_param_keys:
                if key in ['X0', 'Yref', 'D']:
                    res[key] = self.opti.debug.value(self.opti_params[key])
                else:
                    var = self.opti_params[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j] = self.opti.debug.value(var[j])

                    res[key] = values

            u = res['U']
            res['U'] = np.maximum(np.minimum(u, u_max), u_min)
            # print('U_0:', res['U'])
            # print('J:', res['J'])

        return res, feas




class OffsetFreeMPC(MPC):
    """
    This class defines an offset-free MPC implementation. This class utilizaes 
    the Opti stack interface of CasADi.
    """
    def __init__(self, prob_info, offset_style='limon'):
        super(OffsetFreeMPC, self).__init__(prob_info)
        self.mpc_type = 'offsetfree'
        self.offset_style = offset_style

    def get_mpc(self):
        """
        This method creates the optimization problem for the MPC. All 
        information necessary for the creation of this controller is passed 
        upon instantiation of this object within the prob_info dictionary. For 
        more details on the optimization problem, the user is referred to the 
        paper associated with the release of this code.
        
        This code uses IPOPT for the NLP solver which is distributed with 
        CasADi. Users are referred to IPOPT [https://coin-or.github.io/Ipopt/] 
        and the associated paper for more information on this solver.
        """
        ## unpack relevant problem information
        # prediction horizon
        Np = self.prob_info['Np']
        # system dimensions
        nu = self.prob_info['nu']
        nx = self.prob_info['nx']
        ny = self.prob_info['ny']
        nyc = self.prob_info['nyc']
        nd = self.prob_info['nd']
        # state/input/output constraints
        u_min = self.prob_info['u_min']
        u_max = self.prob_info['u_max']
        x_min = self.prob_info['x_min']
        x_max = self.prob_info['x_max']
        y_min = self.prob_info['y_min']
        y_max = self.prob_info['y_max']
        yc_min = self.prob_info['yc_min']
        yc_max = self.prob_info['yc_max']
        # initial guesses for solver
        u_init = self.prob_info['u_init']
        x_init = self.prob_info['x_init']
        y_init = self.prob_info['y_init']
        # dynamic equations (state evolution and output equation)
        f = self.prob_info['f']
        h = self.prob_info['h']
        r = self.prob_info['r']
        # OCP costs
        stage_cost = self.prob_info['stage_cost']
        term_cost = self.prob_info['term_cost']
        # additional input cost/objectives and/or constraints
        reduce_dinput = False
        constrain_dinput = False
        if 'ustage' in self.prob_info.keys():
            ustage = self.prob_info['ustage']
            reduce_dinput = True
        if 'du_max' in self.prob_info.keys():
            du_max = self.prob_info['du_max']
            du_min = self.prob_info['du_min']
            constrain_dinput = True

        # create NLP opti object
        opti = cas.Opti()

        # Initialize container lists for all states, inputs, and outputs
        X = [0 for j in range(Np+1)]
        # Y = [0 for j in range(Np+1)]
        U = [0 for j in range(Np)]

        J = 0 # initialize cost/objective function

        ## define parameter(s), variable(s), and problem
        X[0] = opti.parameter(nx) # initial state as a parameter
        opti.set_value(X[0], np.zeros((nx,1)))

        D = opti.parameter(nd) # offset free disturbances
        opti.set_value(D, np.zeros((nyc,1)))

        # Y[0] = opti.variable(ny) # initial output variable
        # opti.subject_to(Y[0] == h(X[0], D))
        # opti.set_initial(Y[0], y_init)

        if self.offset_style == 'limon':
            # limon style implementation
            Yref = opti.parameter(nyc)
            opti.set_value(Yref, np.zeros((nyc,1)))

            Xss = opti.variable(nx)
            opti.subject_to(opti.bounded(x_min, Xss, x_max))
            opti.set_initial(Xss, x_init)

            Uss = opti.variable(nu)
            opti.subject_to(opti.bounded(u_min, Uss, u_max))
            opti.set_initial(Uss, u_init)

            Yss = opti.variable(ny)
            opti.subject_to(opti.bounded(y_min, Yss, y_max))
            opti.set_initial(Yss, y_init)

            Ycss = opti.variable(nyc)
            opti.subject_to(opti.bounded(yc_min, Ycss, yc_max))
            opti.set_initial(Ycss, np.zeros((nyc,1)))

            opti.subject_to(Xss == f(Xss, Uss, D))
            opti.subject_to(Yss == h(Xss, D))
            opti.subject_to(Ycss == r(Yss))

        # the loop below systematically defines the variables of the optimal
        # control problem (OCP) over the prediction horizon
        for k in range(Np):
            # variable and constraints for u_{k}
            U[k] = opti.variable(nu)
            opti.subject_to(opti.bounded(u_min, U[k], u_max))
            opti.set_initial(U[k], u_init)

            Jstage = stage_cost(X[k], U[k], Xss, Uss)
            J += Jstage # add to the cost (construction of the objective)

            # variable x_{k+1}
            X[k+1] = opti.variable(nx)
            opti.subject_to(opti.bounded(x_min, X[k+1], x_max))
            opti.set_initial(X[k+1], x_init)

            # # variable y_{k+1}
            # Y[k+1] = opti.variable(ny)
            # opti.subject_to(opti.bounded(y_min, Y[k+1], y_max))
            # opti.set_initial(Y[k+1], y_init)

            # dynamics constraint
            opti.subject_to(X[k+1] == f(X[k],U[k],D))

            # # output equality constraint
            # opti.subject_to(Y[k+1] == h(X[k+1],D))

            if k>0 and reduce_dinput:
                J += ustage(U[k],U[k-1])
            elif k>0 and constrain_dinput:
                opti.subject_to(opti.bounded(du_min, U[k]-U[k-1], du_max))

        # terminal cost and constraints
        J_end = term_cost(X[-1], Xss)
        J += J_end

        # target penalty
        if self.offset_style == 'limon':
            if 'target_penalty' in self.prob_info.keys():
                target_penalty = self.prob_info['target_penalty']
            else:
                print('no target penalty defined, assuming 0...')
                target_penalty = 0.0
            J += target_penalty * cas.sumsqr((Yref-Ycss))

        # terminal inequality constraint
        if 'term_eq_cons' in self.prob_info.keys():
            if self.prob_info['term_eq_cons']:
                opti.subject_to(X[-1] == Xss)

        # set to minimize objective/cost
        opti.minimize( J )

        # set solver options
        p_opts = {'verbose': 0,
                  'expand': True,
                  'print_time': 0}          # problem options
        s_opts = {'max_iter': 1000,
                  'print_level': 0,
                  'tol': 1e-6}              # solver options
        opti.solver('ipopt', p_opts, s_opts) # add the solver to the opti object

        # save list containers of variables/parameters into a dict for portability
        opti_vars = {}
        opti_vars['U'] = U
        opti_vars['X'] = X[1:]
        # opti_vars['Y'] = Y
        opti_vars['J'] = J

        opti_params  = {}
        opti_params['X0'] = X[0]
        opti_params['D'] = D

        if self.offset_style == 'limon':
            opti_vars['Xss'] = Xss
            opti_vars['Uss'] = Uss
            opti_vars['Yss'] = Yss 
            opti_vars['Ycss'] = Ycss 

            opti_params['Yref'] = Yref 
        else:
            pass

        # save opti object and variable containers as attributes of NominalMPC
        # object
        self.opti = opti
        self.opti_vars = opti_vars
        self.opti_params = opti_params

        return opti, opti_vars, opti_params
    
    def reset_initial_guesses(self):
        """
        This method resets the initial guesses for the variabls in the optimization problem back to those defined in the prob_info dictionary provided upon instantiation of the OffsetFreeMPC object
        """
        # unpack relevant information from the prob_info dictionary
        Np = self.prob_info['Np']
        nyc = self.prob_info['nyc']
        u_init = self.prob_info['u_init']
        x_init = self.prob_info['x_init']
        y_init = self.prob_info['y_init']

        # unpack relevant variable containers from problem creation
        U = self.opti_vars['U']
        X = self.opti_vars['X']
        # Y = self.opti_vars['Y']
        if self.offset_style == 'limon':
            Uss = self.opti_vars['Uss']
            Xss = self.opti_vars['Xss']
            Yss = self.opti_vars['Yss']
            Ycss = self.opti_vars['Ycss']

            self.opti.set_initial(Uss, u_init)
            self.opti.set_initial(Xss, x_init)
            self.opti.set_initial(Yss, y_init)
            self.opti.set_initial(Ycss, np.zeros((nyc,1)))
        
        for k in range(Np):
            self.opti.set_initial(U[k], u_init)
            self.opti.set_initial(X[k], x_init)
        #     self.opti.set_initial(Y[k], y_init)
        # self.opti.set_initial(Y[k+1], y_init)

    def set_parameters(self, params_list):
        """
        This method sets the values of the parameters of the optimization
        problem to those provided as arguments to this method. The argument
        params_list is a list of new parameter values to set in the same order
        as the opti_param keys.
        """

        # unpack parameter containers
        X0 = self.opti_params['X0']
        D = self.opti_params['D']
        if self.offset_style == 'limon':
            Yref = self.opti_params['Yref']

        # unpack params_list argument
        x0 = params_list[0]
        d = params_list[1]
        if self.offset_style == 'limon':
            yref = params_list[2]

            self.opti.set_value(Yref, yref)

        # reset disturbance
        self.opti.set_value(D, d)

        # reset initial condition
        self.opti.set_value(X0, x0)

    def solve_mpc(self):
        """
        This method solves the MPC as defined by the get_mpc() method of this
        class. This method can only be called after the the get_mpc() method has
        been called (i.e., the optimization problme must be defined before it
        can be solved).
        """
        # extract all keys from the opti variables dict
        opti_var_keys = [*self.opti_vars]
        opti_param_keys = [*self.opti_params]

        # unpack relevant information from problem creation
        u_min = self.prob_info['u_min']
        u_max = self.prob_info['u_max']

        feas = True
        res = {}
        try:
            sol = self.opti.solve()

            for key in opti_var_keys:
                if key in ['J', 'Xss', 'Uss', 'Yss', 'Ycss']:
                    res[key] = sol.value(self.opti_vars[key])
                else:
                    var = self.opti_vars[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j,] = sol.value(var[j])

                    res[key] = values

            res['Ufull'] = res['U']
            res['U'] = res['U'][:,0]

            for key in opti_param_keys:
                if key in ['X0', 'Yref', 'D']:
                    res[key] = sol.value(self.opti_params[key])
                else:
                    var = self.opti_params[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j] = sol.value(var[j])

                    res[key] = values

            if self.prob_info['warm_start']:
                self.opti.set_initial(sol.value_variables())

        except Exception as e:
            print(e)
            # if solve fails, get the last value
            feas = False

            for key in opti_var_keys:
                if key in ['J', 'Xss', 'Uss', 'Yss', 'Ycss']:
                    res[key] = self.opti.debug.value(self.opti_vars[key])
                else:
                    var = self.opti_vars[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j] = self.opti.debug.value(var[j])

                    res[key] = values

            res['Ufull'] = res['U']
            res['U'] = res['U'][:,0]

            for key in opti_param_keys:
                if key in ['X0', 'Yref', 'D']:
                    res[key] = self.opti.debug.value(self.opti_params[key])
                else:
                    var = self.opti_params[key]
                    r = len(var) # Np
                    nx = (var[0]).size1()
                    values = np.zeros((nx,r))
                    for j in range(r):
                        values[:,j] = self.opti.debug.value(var[j])

                    res[key] = values

            u = res['U']
            res['U'] = np.maximum(np.minimum(u, u_max), u_min)
            # print('U_0:', res['U'])
            # print('J:', res['J'])

        return res, feas

# state observers
#
# This script provides definitions of classes that can be used for a state 
# estimator. This script is part of the package of code that produces the 
# results of the following paper:
#
#
# Requirements:
# * Python 3
#
# Copyright (c) 2023 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import numpy as np
import casadi as cas

class StateEstimator():
    """
    StateEstimator is a super class designed to be a template for particular
    implementations of state estimators. Users should develop their own state 
    estimators by using the general structure/methods provided below. Upon
    initialization of this class or any of its child classes, users should
    provide a Python dict that contains all of the relevant problem information.
    """

    def __init__(self, prob_info):
        super(StateEstimator, self).__init__()
        self.prob_info = prob_info

    def get_observer(self):
        """
        This method should generate the state estimator by unpacking relevant
        information from the prob_info dict defined upon instantiation of the
        class.
        """
        pass

    def update_observer(self, u , ymeas):
        """
        This method should update the state estimator by using the arguments 
        provided to this method.
        """
        pass

class EKF(StateEstimator):

    def get_observer(self):
        self.Q = self.prob_info['Qobs']
        self.R = self.prob_info['Robs']

        nx = self.prob_info['nx']
        nu = self.prob_info['nu']
        nd = self.prob_info['nd']

        self.xhat = np.zeros((nx,1))
        self.dhat = np.zeros((nd,1))
        self.P = np.zeros((nx+nd,nx+nd))

        x = cas.SX.sym('x', nx)
        u = cas.SX.sym('u', nu)
        d = cas.SX.sym('d', nd)

        xdot = self.prob_info['f'](x,u,d)
        ymeas = self.prob_info['h'](x,d)
        ddot = d 
        self.A1 = cas.Function('A1', [x,u,d], [cas.jacobian(cas.vertcat(xdot,ddot), cas.vertcat(x,d))])
        self.H1 = cas.Function('H1', [x,d], [cas.jacobian(ymeas, cas.vertcat(x,d))])

    def update_observer(self, u, ymeas):
        u = u.reshape(-1,1)
        ymeas = ymeas.reshape(-1,1)

        # get predicted states
        x_next = (self.prob_info['f'](self.xhat, u, self.dhat)).full()
        d_next = self.dhat

        # get predicted measurement
        y_next = (self.prob_info['h'](x_next, d_next)).full()

        # get predicted covariance
        A = (self.A1(self.xhat, u, self.dhat)).full()
        H = (self.H1(self.xhat, self.dhat)).full()
        phi = A
        P_aug_next = phi @ self.P @ phi.T + self.Q 

        # get kalman gain
        S_next = np.linalg.pinv(H @ P_aug_next @ H.T + self.R)
        K_aug_next = P_aug_next @ H.T @ S_next
        
        # update state estimation and covariance
        x_aug_next = np.vstack((x_next, d_next))
        x_aug_update = x_aug_next + K_aug_next @ (ymeas - y_next)
        P_aug_update = (np.eye(self.prob_info['nx']+self.prob_info['nd']) - K_aug_next @ H) @ P_aug_next

        # update observer properties
        self.xhat = x_aug_update[:self.prob_info['nx']]
        self.dhat = x_aug_update[self.prob_info['nx']:]
        self.P = P_aug_update

        return self.xhat, self.dhat



"""
# **************************<><><><><>*******************************
# * Script to Provide Models & Dynamics for AEKF Python Translation *
# **************************<><><><><>*******************************
#
# This script and all its dependencies are implemented by: Esmaeil Seraj
#   - Esmaeil Seraj, CORE Robotics Lab, Robotics & Intelligent Machines,
#   Georgia Tech, Atlanta, GA, USA
#   - email <eseraj3@gatech.edu>
#   - website <https://github.gatech.edu/MCG-Lab/DistributedControl>
#
# Published under GNU GENERAL PUBLIC LICENSE ver. 3 (or any later version)
#
"""

import numpy as np


# wildfire model dynamics for EKF
class ModelDynamicsAEKF(object):

    def __init__(self, F_dim, H_dim, time_step=1):
        self.time_step = time_step
        self.n = F_dim
        self.m = H_dim

    # initialize parameters for EKF
    def kf_initializer(self, qx, qy, px, py, pz, R, U, Theta):
        """
        This function initializes the parameters for AEKF

        :param qx: x coordination of current firer spot
        :param qy: y coordination of current firer spot
        :param px: x coordination of current drone's pose
        :param py: y coordination of current drone's pose
        :param pz: z coordination of current drone's pose
        :param R: fuel coefficient
        :param U: average mid-flame wind speed
        :param Theta: wind azimuth
        :return: initial state and covariance estimates
        """

        rand_val = np.array(
            [[np.random.randn()], [np.random.randn()], [np.random.rand()], [np.random.rand()], [np.random.rand()],
             [0], [0], [0]])
        x0 = np.array([[qx], [qy], [px], [py], [pz], [R], [U], [Theta]]) + rand_val
        P0 = np.zeros(shape=[self.n, self.n])

        # state covariance matrix
        Q0 = 1e-8 * np.eye(self.n)  # greater Q means more reliance on observations
        Q0[0, 1], Q0[1, 0] = np.finfo(float).eps, np.finfo(float).eps
        Q0[2, 4], Q0[2, 3] = np.finfo(float).eps, np.finfo(float).eps
        Q0[3, 4], Q0[3, 2] = np.finfo(float).eps, np.finfo(float).eps
        Q0[4, 2], Q0[4, 3] = np.finfo(float).eps, np.finfo(float).eps

        # measurement (observation) covariance matrix
        R0 = 1e-1 * np.eye(self.m)  # smaller R:: more reliance on observations, Greater R:: longer convergence
        R0[0, 1], R0[1, 0] = np.finfo(float).eps, np.finfo(float).eps

        return x0, P0, Q0, R0

    # calculate gradients for state transition Jacobian matrix
    def state_gradients(self, R, U, Theta):
        """
        This function calculates the gradients for state transition Jacobian matrix F

        :param R: fuel coefficient
        :param U: average mid-flame wind speed
        :param Theta: wind azimuth
        :return: a dictionary including all state-transition Jacobian matrix gradients
        """

        # calculating fire dynamics
        a, b, c, d, ll = 0.936, 0.2566, 0.461, 0.1548, - 0.397  # fire model constants
        LB = a * np.exp(b * U) + c * np.exp(-d * U) + ll
        GB = np.absolute(LB ** 2 - 1)
        C_UR = R * (1 - (LB / (LB + np.sqrt(GB))))

        qxdot = C_UR * np.sin(Theta)
        qydot = C_UR * np.cos(Theta)

        # state transition gradients
        dLB_dU = a * b * np.exp(b * U) - c * d * np.exp(-d * U)
        dGB_dU = 2 * (a ** 2) * b * np.exp(2 * b * U) - 2 * (c ** 2) * d * np.exp(-2 * d * U) + a * (b - d) * np.exp(
            (b - d) * U) - 2 * ll * dLB_dU

        dC_UR_dR = 1 - (LB / (LB + np.sqrt(GB)))
        dC_UR_dU = (R * ((LB * dGB_dU) - (GB * dLB_dU))) / (LB + np.sqrt(GB)) ** 2

        dqxdot_dR = dC_UR_dR * np.sin(Theta)
        dqydot_dR = dC_UR_dR * np.cos(Theta)
        dqxdot_dU = dC_UR_dU * np.sin(Theta)
        dqydot_dU = dC_UR_dU * np.cos(Theta)
        dqxdot_dTheta = C_UR * np.cos(Theta)
        dqydot_dTheta = -C_UR * np.sin(Theta)

        dqx_dR = dqxdot_dR * self.time_step
        dqx_dU = dqxdot_dU * self.time_step
        dqx_dTheta = dqxdot_dTheta * self.time_step

        dqy_dR = dqydot_dR * self.time_step
        dqy_dU = dqydot_dU * self.time_step
        dqy_dTheta = dqydot_dTheta * self.time_step

        state_grads = {'qxdot': qxdot,
                       'qydot': qydot,
                       'dqx_dR': dqx_dR,
                       'dqx_dU': dqx_dU,
                       'dqx_dTheta': dqx_dTheta,
                       'dqy_dR': dqy_dR,
                       'dqy_dU': dqy_dU,
                       'dqy_dTheta': dqy_dTheta}

        return state_grads

    # calculate gradients for observation Jacobian matrix
    @staticmethod
    def observation_gradients(qx, qy, px, py, pz):
        """
        This function calculates the gradients for observation Jacobian matrix H

        :param qx: x coordination of current firer spot
        :param qy: y coordination of current firer spot
        :param px: x coordination of current drone's pose
        :param py: y coordination of current drone's pose
        :param pz: z coordination of current drone's pose
        :return:  a dictionary including all observation Jacobian matrix gradients
        """

        # observation gradients
        dphix_rem = 1 / (1 + ((qx - px) / pz) ** 2)
        dphiy_rem = 1 / (1 + ((qy - py) / pz) ** 2)

        dphix_dqx = dphix_rem * (1 / pz)
        dphix_dpx = dphix_rem * (-1 / pz)
        dphix_dpz = dphix_rem * (qx - pz) * (-1 / pz ** 2)

        dphiy_dqy = dphiy_rem * (1 / pz)
        dphiy_dpy = dphiy_rem * (-1 / pz)
        dphiy_dpz = dphiy_rem * (qy - pz) * (-1 / pz ** 2)

        observation_grads = {'dphix_dqx': dphix_dqx,
                             'dphix_dpx': dphix_dpx,
                             'dphix_dpz': dphix_dpz,
                             'dphiy_dqy': dphiy_dqy,
                             'dphiy_dpy': dphiy_dpy,
                             'dphiy_dpz': dphiy_dpz}

        return observation_grads

    # form the state transition Jacobian matrix
    @staticmethod
    def state_jacobian(state_grads):
        """
        This function calculates and forms the state-transition Jacobian matrix F

        :param state_grads: a dictionary including all gradients [output of state_gradients()]
        :return: state-transition Jacobian matrix F
        """

        # extracting the gradients
        dqx_dR = state_grads['dqx_dR']
        dqx_dU = state_grads['dqx_dU']
        dqx_dTheta = state_grads['dqx_dTheta']

        dqy_dR = state_grads['dqy_dR']
        dqy_dU = state_grads['dqy_dU']
        dqy_dTheta = state_grads['dqy_dTheta']

        # forming the Jacobian matrix
        F = np.array([[1, 0, 0, 0, 0, dqx_dR, dqx_dU, dqx_dTheta],
                      [0, 1, 0, 0, 0, dqy_dR, dqy_dU, dqy_dTheta],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])

        return F

    # form the observation Jacobian matrix
    def observation_jacobian(self, observation_grads, mapping_flg):
        """
        This function calculates and forms the observation Jacobian matrix H

        :param observation_grads: a dictionary including all gradients [output of observation_gradients()]
        :param mapping_flg: boolean flag showing is there gonna be any dynamics (e.g., mapping) for observation model?
        :return: observation Jacobian matrix H
        """

        # check if mapping is required
        if not mapping_flg:
            H = np.eye(self.m)
        else:
            # extracting the gradients
            dphix_dqx = observation_grads['dphix_dqx']
            dphix_dpx = observation_grads['dphix_dpx']
            dphix_dpz = observation_grads['dphix_dpz']

            dphiy_dqy = observation_grads['dphiy_dqy']
            dphiy_dpy = observation_grads['dphiy_dpy']
            dphiy_dpz = observation_grads['dphiy_dpz']

            # forming the Jacobian matrix
            H = np.array([[dphix_dqx, 0, dphix_dpx, 0, dphix_dpz, 0, 0, 0],
                          [0, dphiy_dqy, 0, dphiy_dpy, dphiy_dpz, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])

        return H

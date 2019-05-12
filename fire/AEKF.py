"""
# *************************<><><><><>*******************************
# * Script for Adaptive Extended Kalman Filter Python  Translation *
# *************************<><><><><>*******************************
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


# AEKF
class AEKF(object):

    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F  # process Jacobian matrix
        self.H = H  # observation Jacobian matrix
        self.B = 0 if B is None else B  # input command transition matrix
        self.Q = np.eye(self.n) if Q is None else Q  # process noise covariance matrix
        self.R = np.eye(self.n) if R is None else R  # observation noise covariance matrix
        self.P = np.eye(self.n) if P is None else P  # prediction covariance
        self.x = np.zeros((self.n, 1)) if x0 is None else x0  # initial state estimate

    # AEKF prediction step
    def predict(self, u=0):
        """
        This function implements the prediction step for EKF

        :param u: input command vector [default:: u = 0]
        :return: state estimate x
        """

        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x

    # AEKF update step
    def update(self, z, adaptive_flg=False, forgetting_factor=0.1):
        """
        This function implements the update step for EKF [or AEKF (optional)]

        :param z: observation vector
        :param adaptive_flg: boolean flag which switches EKF to AEKF and vice versa [default:: adaptive_flg=False]
        :param forgetting_factor: in case of AEKF, this scalar specifies the forgetting factor for learning Q and R
        :return: the covariance residual (as a measure of uncertainty)
        """

        y = z - np.dot(self.H, self.x)  # innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # covariance residual
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # measurement update (correction)
        I = np.eye(self.n)

        # covariance updates
        self.P = \
            np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        if adaptive_flg:
            alpha = forgetting_factor
            y_pred = np.dot(self.H, self.x)  # observation prediction
            residual = z - y_pred  # residual
            # process noise covariance update
            self.Q = alpha * self.Q + (1 - alpha) * np.dot(np.dot(K, np.dot(y, y.T)), K.T)
            # measurement noise covariance update
            self.R = \
                alpha * self.R + (1 - alpha) * (np.dot(residual, residual.T) + np.dot(np.dot(self.H, self.P), self.H.T))

        return S

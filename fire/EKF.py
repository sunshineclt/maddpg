"""
# ********************<><><><><>**************************
# * Script for Extended Kalman Filter Python Translation *
# ********************<><><><><>**************************
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


# EKF
class EKF(object):

    def __init__(self, F=None, H=None, xhat0=None, P0=None, Q=None, R=None, num_iter=200):
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[0]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        self.num_iter = num_iter  # number of iterations for prediction
        self.xhat0 = np.zeros((self.n, 1)) if xhat0 is None else xhat0  # initial guess for state estimate
        self.P0 = np.eye(self.n) if P0 is None else P0  # initial guess for prediction covariance
        self.Q = 1e-8*np.eye(self.n) if Q is None else Q  # process variance (greater Q:: more reliance on observations)
        self.R = 1e-3*np.eye(self.n) if R is None else R  # measurement variance (smaller R: more trust on observations)

    # kalman filter
    def ekf(self, z=None, measurement_quality=1):
        """
        This function implements the Extended Kalman filter

        :param z: vector of noisy observations
        :param measurement_quality: variance of the noise (higher means worse quality) [default:: 1]
        :return: state prediction and uncertainty residual for state estimate
        """

        # initialize parameters
        sz = (self.num_iter, self.n)  # size of array [rows are observations and  columns are states]
        xhat = np.zeros(sz)  # a posteriori estimate of x
        xhatminus = np.zeros(sz)  # a priori estimate of x
        P = []  # a posteriori error estimate
        Pminus = []  # a priori error estimate
        K = []  # Kalman gain
        xhat[0, :] = self.xhat0[:, 0]  # initial guess for state estimate
        P.append(self.P0)  # initial guess for prediction covariance
        Pminus.append(self.P0)
        K.append(np.zeros(shape=[self.n, self.m]))

        # generate noisy observations if none provided
        z = self.xhat0[:, 0] + np.random.normal(0, measurement_quality, size=sz) if z is None else z
        assert int(z.shape[1]) == int(self.m), "In case obs-space and state-space dimensions do not match, z has to be provided!!"

        # kalman predictor
        for obs in range(1, self.num_iter):
            # Kalman filter prediction step
            xhatminus[obs, :] = np.dot(self.F, xhat[obs - 1, :])
            Pminus.append(np.dot(np.dot(self.F, P[obs - 1]), self.F.T) + self.Q)

            # Kalman filter update step
            K.append(np.dot(np.dot(Pminus[obs], self.H.T), np.linalg.inv(np.dot(self.H, np.dot(Pminus[obs], self.H.T)) + self.R)))
            xhat[obs, :] = xhatminus[obs, :] + np.dot(K[obs], (z[obs, :] - xhatminus[obs, :]))
            P.append(np.dot((np.eye(self.n) - np.dot(K[obs], self.H)), Pminus[obs]))

        error_residuals = np.dot(self.H, np.dot(Pminus[-1], self.H.T)) + self.R
        state_predictions = xhat[-1, :]

        return state_predictions, np.diag(error_residuals)

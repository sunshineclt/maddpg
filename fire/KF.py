"""
# ****************<><><><><>**********************
# * Script for Kalman Filter Python  Translation *
# ****************<><><><><>**********************
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


# KF
class KF(object):

    def __init__(self, xhat0=None, P0=None, Q=None, R=None, num_iter=200):
        self.num_iter = num_iter  # number of iterations for prediction
        self.xhat0 = 0.0 if xhat0 is None else xhat0  # initial guess for state estimate
        self.P0 = 1.0 if P0 is None else P0  # initial guess for prediction covariance
        self.Q = 1e-8 if Q is None else Q  # process variance (greater Q:: more reliance on observations)
        self.R = 1e-3 if R is None else R  # measurement variance (smaller R:: more reliance on observations)

    # kalman filter
    def kf(self, x, z=None):
        """
        This function implements the Kalman filter

        :param x: state variable
        :param z: vector of noisy observations
        :return: state prediction and uncertainty residual for state estimate
        """

        # initialize parameters
        sz = (self.num_iter,)  # size of array
        xhat = np.zeros(sz)  # a posteriori estimate of x
        P = np.zeros(sz)  # a posteriori error estimate
        xhatminus = np.zeros(sz)  # a priori estimate of x
        Pminus = np.zeros(sz)  # a priori error estimate
        K = np.zeros(sz)  # Kalman gain
        xhat[0] = self.xhat0  # initial guess for state estimate
        P[0] = self.P0  # initial guess for prediction covariance

        # generate noisy observations if none provided
        z = np.random.normal(x, 1, size=sz) if z is None else z

        # kalman predictor
        for obs in range(1, self.num_iter):
            # Kalman filter prediction step
            xhatminus[obs] = xhat[obs - 1]
            Pminus[obs] = P[obs - 1] + self.Q

            # Kalman filter update step
            K[obs] = Pminus[obs] / (Pminus[obs] + self.R)
            xhat[obs] = xhatminus[obs] + K[obs] * (z[obs] - xhatminus[obs])
            P[obs] = (1 - K[obs]) * Pminus[obs]

        error_residual = Pminus[-1] + self.R
        state_prediction = xhat[-1]

        return state_prediction, error_residual

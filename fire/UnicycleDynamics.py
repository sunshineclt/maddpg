"""
# ********************<><><><><>***************************
# * Script for Robot Unicycle Dynamics Python Translation *
# ********************<><><><><>***************************
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


# Unicycle Dynamics
class UnicycleDynamics(object):

    def __init__(self, time_step=1, pos_error=1, rot_error=100, prop_ctrl_gain=1, prop_controller_mag_limit=1,
                 linear_velocity_limit=1, angular_velocity_limit=4*np.pi):
        self.time_step = time_step
        self.position_error = pos_error
        self.rotation_error = rot_error  # is set to 100 because we don't care about it here
        self.prop_controller_gain = prop_ctrl_gain  # proportional gain for controller
        self.prop_controller_mag_limit = prop_controller_mag_limit  # magnitude limit of the produced single-integrator ctrl input
        self.linear_velocity_limit = linear_velocity_limit
        self.angular_velocity_limit = angular_velocity_limit

    # single integrator position controller [currently just a proportional controller]
    def single_integrator_position_controller(self, x, poses):
        """
        This function implements a position controller for single integrators.  Drives a single integrator to a point
        using a proportional controller.

        :param x: 2xNum_agents numpy array of current single-integrator states
        :param poses: 2xNum_agents numpy array of desired poses
        :return: 2xNum_agents numpy array of single-integrator control inputs
        """

        # calculate the proportional control input
        dxi = self.prop_controller_gain * (poses - x)

        # TODO: if any other type of controller is needed (e.g., PID) here would be to define the control law

        # threshold magnitude
        norms = np.linalg.norm(dxi, axis=0)
        idxs = np.where(norms > self.prop_controller_mag_limit)
        dxi[:, idxs] *= self.prop_controller_mag_limit / norms[idxs]

        return dxi

    # map from single integrator to unicycle
    def single_integrator_to_unicycle(self, dxi, poses):
        """
        This function maps control inputs from single-integrator to unicycle dynamics

        :param dxi: 2xNum_agents numpy array of single-integrator control inputs [output of single_integrator_position_controller]
        :param poses: 2xNum_agents numpy array of desired poses
        :return: 2xNum_agents numpy array of unicycle control inputs
        """

        m, n = np.shape(dxi)

        a = np.cos(poses[2, :])
        b = np.sin(poses[2, :])

        dxu = np.zeros((2, n))
        dxu[0, :] = self.linear_velocity_limit * (a * dxi[0, :] + b * dxi[1, :])
        dxu[1, :] = (np.pi / 2) * self.angular_velocity_limit * np.arctan2(-b * dxi[0, :] + a * dxi[1, :], dxu[0, :])

        return dxu

    # direct pose controller for unicycle models
    def unicycle_pose_controller(self, x, poses):
        """
        This function implements a pose controller for unicycle models.  This is a hybrid controller that first
        drives the unicycle to a point then turns the unicycle to match the orientation.

        :param x: 3xNum_agents numpy array of unicycle states
        :param poses: 3xNum_agents numpy array of desired poses
        :return: 2xNum_agents numpy array of unicycle control inputs
        """

        _, n = np.shape(x)
        dxu = np.zeros((2, n))

        # get the norms
        norms = np.linalg.norm(poses[:2, :] - x[:2, :], axis=0)

        # figure out who's close enough
        not_there = np.where(norms > self.position_error)[0]
        there = np.where(norms <= self.position_error)[0]

        # calculate angle proportional controller
        wrapped_angles = poses[2, there] - x[2, there]
        wrapped_angles = np.arctan2(np.sin(wrapped_angles), np.cos(wrapped_angles))

        # get a proportional controller for position
        dxi = single_integrator_position_controller(x[:2, :], poses[:2, :])  # TODO: WTF is wrong with these?

        # decide what to do based on how close we are
        dxu[:, not_there] = single_integrator_to_unicycle(dxi[:, not_there], x[:, not_there])  # TODO: WTF is wrong with these?
        dxu[:, there] = np.vstack([np.zeros(np.size(there)), wrapped_angles])

        return dxu

    # map from unicycle to single integrator
    @staticmethod
    def unicycle_to_single_integrator(dxu, poses, projection_distance=0.05):
        """
        This function converts from unicycle to single-integrator dynamics through a virtual point placed in front of the unicycle

        :param dxu: 2xNum_agents numpy array of unicycle control inputs
        :param poses: 3xNum_agents numpy array of desired poses
        :param projection_distance: how far ahead of the unicycle model to place the virtual point
        :return: 2xNum_agents numpy array of single-integrator control inputs
        """

        m, n = np.shape(dxu)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxi = np.zeros((2, n))
        dxi[0, :] = (cs*dxu[0, :] - projection_distance*ss*dxu[1, :])
        dxi[1, :] = (ss*dxu[0, :] + projection_distance*cs*dxu[1, :])

        return dxi

    # check if conditions are satisfied for agents or not
    def at_pose(self, states, poses):
        """
        This function checks whether agents are "close enough" to required poses or not

        :param states: 3xNum_agents numpy array of unicycle states
        :param poses: 3xNum_agents numpy array of desired states
        :return: 1xNum_agents numpy index array of agents that are close enough
        """

        # calculate rotation errors with angle wrapping
        res = states[2, :] - poses[2, :]
        res = np.abs(np.arctan2(np.sin(res), np.cos(res)))

        # calculate position errors
        pes = np.linalg.norm(states[:2, :] - poses[:2, :], 2, 0)

        # determine which agents are done
        done = np.nonzero((res <= self.rotation_error) & (pes <= self.position_error))

        return done
